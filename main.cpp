// ============================================================================
// Домашнее задание: MPI-паттерны
// Паттерн №7: «Коллективные операции как основной механизм»
// Задача: статистика большого вектора (среднее, дисперсия, максимум)
//         и скалярное произведение двух векторов.
//
// Идея паттерна: вместо ручных циклов "for (r...) MPI_Send/MPI_Recv" весь обмен
// данными выполняется коллективными вызовами, в которых участвуют ВСЕ процессы:
//   MPI_Bcast     - раздать всем параметры задачи (размер N);
//   MPI_Scatterv  - раздать куски массивов (N не обязано делиться на size);
//   MPI_Allreduce - получить среднее на ВСЕХ процессах (нужно для дисперсии);
//   MPI_Reduce    - собрать итоговые суммы/максимум на ранге 0;
//   MPI_Gather    - собрать на ранге 0 размеры кусков (для отчёта).
// Ни одного MPI_Send / MPI_Recv в программе нет.
// ============================================================================
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace std;   // чтобы не писать std:: перед vector, max, sin, printf и т.д.

static const int STUDENT_NUMBER = 7;                 // номер N в eclass
static const char* PATTERN = "7: коллективные операции (Bcast/Scatterv/Allreduce/Reduce/Gather)";
static const char* TASK    = "среднее, дисперсия, max и скалярное произведение вектора";

// Детерминированная генерация данных (одинакова для послед. и паралл. версий)
static void generate(vector<double>& x, vector<double>& y, long n) {
    x.resize(n); y.resize(n);
    for (long i = 0; i < n; ++i) {
        x[i] = sin(i * 1e-3) + (i % 100) * 0.01;
        y[i] = cos(i * 1e-3) * 0.5 + (i % 7) * 0.1;
    }
}

// Результат вычислений
struct Result { double mean, var, maxv, dot; };

// ---------------------- ПОСЛЕДОВАТЕЛЬНАЯ ВЕРСИЯ -----------------------------
static Result sequential(const vector<double>& x, const vector<double>& y) {
    long n = x.size();
    double sum = 0, mx = x[0], dot = 0;
    for (long i = 0; i < n; ++i) {
        sum += x[i];
        mx = max(mx, x[i]);
        dot += x[i] * y[i];
    }
    double mean = sum / n, sq = 0;
    for (long i = 0; i < n; ++i) sq += (x[i] - mean) * (x[i] - mean);
    return {mean, sq / n, mx, dot};
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long n = 0;
    vector<double> x, y;     // полные данные существуют только на ранге 0
    Result seq{};
    double t_seq = 0;

    // ---- Ранг 0: генерация данных и последовательный расчёт (для сравнения) ----
    if (rank == 0) {
        n = 10000000;             // фиксированный размер задачи
        generate(x, y, n);
        double t0 = MPI_Wtime();
        seq = sequential(x, y);
        t_seq = MPI_Wtime() - t0;
    }

    MPI_Barrier(MPI_COMM_WORLD);  // чтобы замер начался одновременно у всех
    double t0 = MPI_Wtime();

    // ---- 1. Bcast: все процессы узнают размер задачи ----
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // ---- 2. Статическое разбиение: counts[r], displs[r] (одинаково на всех) ----
    vector<int> counts(size), displs(size);
    for (int r = 0, off = 0; r < size; ++r) {
        counts[r] = n / size + (r < n % size ? 1 : 0);   // остаток раздаём первым
        displs[r] = off;
        off += counts[r];
    }
    int local_n = counts[rank];

    // ---- 3. Scatterv: раздаём куски x и y ----
    vector<double> lx(local_n), ly(local_n);
    MPI_Scatterv(x.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 lx.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 ly.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- 4. Локальные суммы, максимум и скалярное произведение ----
    double loc[2] = {0.0, 0.0};          // loc[0] = sum(x), loc[1] = dot(x,y)
    double loc_max = lx[0];              // (local_n >= 1, т.к. n >> size)
    for (int i = 0; i < local_n; ++i) {
        loc[0] += lx[i];
        loc[1] += lx[i] * ly[i];
        loc_max = max(loc_max, lx[i]);
    }

    // ---- 5. Allreduce: глобальная сумма и dot у ВСЕХ процессов ----
    //         (среднее нужно каждому для расчёта дисперсии)
    double glob[2];
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mean = glob[0] / n;

    // ---- 6. Локальная сумма квадратов отклонений ----
    double loc_sq = 0;
    for (int i = 0; i < local_n; ++i) loc_sq += (lx[i] - mean) * (lx[i] - mean);

    // ---- 7. Reduce: итоговые значения на ранге 0 ----
    double glob_sq = 0, glob_max = 0;
    MPI_Reduce(&loc_sq,  &glob_sq,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&loc_max, &glob_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- 8. Gather: размеры кусков на ранг 0 (для отчёта о балансе) ----
    vector<int> all_n(size);
    MPI_Gather(&local_n, 1, MPI_INT, all_n.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Время параллельной части = максимум по процессам
    double t_loc = MPI_Wtime() - t0, t_par = 0;
    MPI_Reduce(&t_loc, &t_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Вывод ----
    if (rank == 0) {
        Result par{mean, glob_sq / n, glob_max, glob[1]};
        auto close = [](double a, double b) {          // порядок суммирования другой
            return fabs(a - b) <= 1e-9 * max(1.0, fabs(b)); };
        bool ok = close(par.mean, seq.mean) && close(par.var, seq.var) &&
                  close(par.maxv, seq.maxv) && close(par.dot, seq.dot);

        printf("Студент №%d | процессов: %d | N = %ld\n", STUDENT_NUMBER, size, n);
        printf("Паттерн: %s\n", PATTERN);
        printf("Задача:  %s\n", TASK);
        printf("Размеры кусков (Gather):");
        for (int r = 0; r < size; ++r) printf(" %d", all_n[r]);
        printf("\n");
        printf("MPI:           mean=%.10f var=%.10f max=%.6f dot=%.6f\n",
                    par.mean, par.var, par.maxv, par.dot);
        printf("Последоват.:   mean=%.10f var=%.10f max=%.6f dot=%.6f\n",
                    seq.mean, seq.var, seq.maxv, seq.dot);
        printf("Результаты совпадают: %s\n", ok ? "ДА" : "НЕТ");
        printf("Время послед.: %.4f с | Время MPI: %.4f с | Ускорение: %.2fx\n",
                    t_seq, t_par, t_seq / t_par);
    }
    MPI_Finalize();
    return 0;
}

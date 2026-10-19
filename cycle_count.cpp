
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cerrno>

#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>

int main() {
    perf_event_attr attr{};
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = PERF_COUNT_HW_CPU_CYCLES;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    int fd = syscall(
        __NR_perf_event_open, &attr, 0, -1, -1, 0
    );

    if (fd == -1) {
        std::cerr << "perf_event_open failed: "
                  << strerror(errno) << '\n';
        return 1;
    }

    const int dimension = 128;
    const int repetitions = 10000;

    std::vector<float> a(dimension, 1.0f);
    std::vector<float> b(dimension, 2.0f);

    // Reset and start counting.
    if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1 ||
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
        perror("Counter start");
        close(fd);
        return 1;
    }

    volatile float result = 0.0f;

    // Perform the distance computations.
    for (int r = 0; r < repetitions; ++r) {
        float sum = 0.0f;

        for (int i = 0; i < dimension; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        result = result + sum;
    }

    // Stop counting BEFORE reading the final value.
    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) == -1) {
        perror("Counter stop");
        close(fd);
        return 1;
    }

    uint64_t total_cycles = 0;

    if (read(fd, &total_cycles, sizeof(total_cycles))
        != sizeof(total_cycles)) {
        perror("Counter read");
        close(fd);
        return 1;
    }

    std::cout << "Total CPU cycles: "
              << total_cycles << '\n';

    std::cout << "Distance computations: "
              << repetitions << '\n';

    std::cout << "Average cycles per computation: "
              << static_cast<double>(total_cycles) / repetitions
              << '\n';

    std::cout << "Result check: " << result << '\n';

    close(fd);
    return 0;
}

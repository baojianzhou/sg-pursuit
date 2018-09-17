//
// Created by baojian on 8/23/18.
//

#include "get_cpu.h"


#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<stdio.h>
#include <stdbool.h>

bool get_cpu_info() {
    char str[100];
    const char d[2] = " ";
    char *token;
    int i = 0, times, lag;
    long int sum = 0, idle, lastSum = 0, lastIdle = 0;
    times = atoi("1");
    lag = atoi("1");
    while (times > 0) {
        FILE *fp = fopen("/proc/stat", "r");
        i = 0;
        fgets(str, 100, fp);
        fclose(fp);
        token = strtok(str, d);
        while (token != NULL) {
            token = strtok(NULL, d);
            if (token != NULL) {
                sum += atoi(token);
                if (i == 3)
                    idle = atoi(token);
                i++;
            }
        }
        printf("\nIdle for : %lf %% of the time.",
               (1.0 - (idle - lastIdle) * 1.0 / (sum - lastSum)) * 100);
        lastIdle = idle;
        lastSum = sum;
        times--;
        sleep(lag);
    }
    return true;
}

int main(int argC, char *argV[]) {
    get_cpu_info();
    return 0;
}
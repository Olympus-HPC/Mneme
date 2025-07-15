//==============================================================================

void mtx_header(const char* file_path,
                int*        num_lines,
                int*        num_rows,
                int*        num_cols,
                int*        nnz,
                int*        is_symmetric) {
    char buffer[256];
    FILE* file = fopen(file_path, "r");
    if (file == NULL) {
        printf("Error: unable to open the file %s\n", file_path);
        exit(EXIT_FAILURE);
    }
    fgets(buffer, 256, file); // skip comments
    char *token = strtok(buffer, " ");
    if (strcmp(token, "%%MatrixMarket") != 0) {
        printf("Unsupported file format. Only MTX format is supported");
        exit(EXIT_FAILURE);
    }
    strtok(NULL, " "); // skip word
    strtok(NULL, " "); // skip word
    token = strtok(NULL, " "); // check data type
    if (strcmp(token, "real") != 0) {
        printf("Only real (double) matrices are supported");
        exit(EXIT_FAILURE);
    }
    token = strtok(NULL, " \n"); // symmetric, unsymmetric
    *is_symmetric = (strcmp(token, "symmetric") == 0);
    while (fgetc(file) == '%')
        fgets(buffer, 256, file); // skip % comments
    fseek(file, -1, SEEK_CUR);
    fscanf(file, "%d %d %d", num_rows, num_cols, num_lines);
    *nnz = (*is_symmetric) ? *num_lines * 2 : *num_lines;
    fclose(file);
}

typedef struct IdxType {
    int    row, col;
    double val;
} Idx;

int sort_by_row(const void *a, const void *b) {
    return ((Idx*) a)->row - ((Idx*) b)->row;
}

void mtx_parsing(const char* file_path,
                 int         num_lines,
                 int         num_rows,
                 int         nnz,
                 int*        rows_offsets,
                 int*        columns,
                 double*     values,
                 int         base) {
    char buffer[256];
    FILE* file = fopen(file_path, "r");
    while (fgetc(file) == '%')
        fgets(buffer, 256, file); // skip comments
    fgets(buffer, 256, file);     // skip num row, cols, nnz

    Idx* idx_tmp = (Idx*) malloc(nnz * sizeof(Idx));
    for (int i = 0; i < num_lines; i++) {
        int    row, column;
        double value;
        fscanf(file, "%d %d %lf ", &row, &column, &value);
        row         -= (1 - base);
        column      -= (1 - base);
        idx_tmp[i].row = row;
        idx_tmp[i].col = column;
        idx_tmp[i].val = value;
        if (nnz != num_lines) { // is stored symmetric
            idx_tmp[i + num_lines].row = column;
            idx_tmp[i + num_lines].col = row;
            idx_tmp[i + num_lines].val = value;
        }
    }
    qsort(idx_tmp, nnz, sizeof(Idx), sort_by_row); // sort by row
    memset(rows_offsets, 0x0, (num_rows + 1) * sizeof(int));
    for (int i = 0; i < nnz; i++)
        rows_offsets[idx_tmp[i].row + 1]++;
    // prefix-scan
    for (int i = 1; i <= num_rows; i++)
        rows_offsets[i] = rows_offsets[i] + rows_offsets[i - 1];
    for (int i = 0; i < nnz; i++) {
        columns[i] = idx_tmp[i].col;
        values[i]  = idx_tmp[i].val;
    }
    fclose(file);
    free(idx_tmp);
}

//==============================================================================


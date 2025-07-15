#include <stdio.h>
#include <stdlib.h>
#define BUFFER_MAXSIZE 1048576

void getmatrixsize(FILE *in,int *nn,int *DD)
{
  int n=0,D=0;
  char *line = NULL,column;

  do {
    column = fgetc (in);
    if (column == '\t') D++;
  } while (column != '\n');

  line=(char *)malloc(BUFFER_MAXSIZE*sizeof(char)); 
  if (!line)
    exit(-1);

  while (fgets (line,BUFFER_MAXSIZE,in)!=NULL && sizeof(line)>1){
    n++;
  }  

  free(line);

  *nn=n;
  *DD=D;

  if (in) fclose(in);
}

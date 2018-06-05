#include "cfortran.h"

PROTOCCALLSFSUB2(INIP,inip,INTV,DOUBLEV)
PROTOCCALLSFSUB4(EVALF,evalf,INT,DOUBLEV,DOUBLEV,INTV)
PROTOCCALLSFSUB4(EVALG,evalg,INT,DOUBLEV,DOUBLEV,INTV)
PROTOCCALLSFSUB3(PROJ,proj,INT,DOUBLEV,INTV)

#define inip(n,x) \
CCALLSFSUB2(INIP,inip,INTV,DOUBLEV,n,x)

#define evalf(n,x,f,flag) \
CCALLSFSUB4(EVALF,evalf,INT,DOUBLEV,DOUBLEV,INTV,n,x,f,flag)

#define evalg(n,x,g, flag) \
CCALLSFSUB4(EVALG,evalg,INT,DOUBLEV,DOUBLEV,INTV,n,x,g,flag)

#define proj(n,x, flag) \
CCALLSFSUB3(PROJ,proj,INT,DOUBLEV,INTV,n,x,flag)


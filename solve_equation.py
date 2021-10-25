#!/usr/bin/env python3
import numpy as np
import sys

argc=len(sys.argv)

if argc<7:
    print("usage: %s a1 b1 c1 a2 b2 c2" % sys.argv[0])
    exit(-1)

P=np.array([
        [int(sys.argv[1]),int(sys.argv[2])],
        [int(sys.argv[4]),int(sys.argv[5])]
])

C=np.array([int(sys.argv[3]),int(sys.argv[6])])

def k(val):
    return val if val != 1.0 else ""

print("{}x+{}y={}\n" \
    "{}x+{}y={}".format(k(P[0][0]),k(P[0][1]),C[0],
        k(P[1][0]),k(P[1][1]),C[1]))

if np.linalg.det(P)==0:
    if P[0][0]/P[1][0]==P[0][1]/P[1][1]==C[0]/C[1]:
        print("Infinite many solutions")
    else:
        print("No solution")
    sys.exit(1)

X=np.linalg.inv(P).dot(C)
print("x={}, y={}".format(X[0],X[1]))

a=int(input())
b=list(map(int,input().split()))
c=int(input())
d=0
if c == 0:
    print(*b)
if c <= -1:
    d = abs(c)
    i=1
    while i <= d:
        b = b[1:]+[b[0]]
        i+=1
    print(*b)
if c >= 1:
    i=1
    while i <= c:
        b = [b[-1]] + b[:-1]
        i+=1
    print(*b)

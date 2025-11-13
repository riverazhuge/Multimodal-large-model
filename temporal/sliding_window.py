def sliding_slices(T:int, win:int=16, overlap:int=4):
    if win >= T:
        return [(0, T)]
    s = 0; out=[]
    while s < T:
        e = min(T, s+win)
        out.append((s,e))
        if e == T: break
        s = e - overlap
    return out

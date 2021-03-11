function [ error ] = errorfunc( P ) 
src = evalin('base', 'A');
dst = evalin('base', 'B');

n = size(src);
error = 0;
for i = 1 : n(1)
    sr = src(i,1);
    sg = src(i,2);
    sb = src(i,3);
    
    dr = dst(i,1);
    dg = dst(i,2);
    db = dst(i,3);

    r = P(1) * sr + P(2) * sg + P(3) * sb;
    g = P(4) * sr + P(5) * sg + P(6) * sb;
    b = P(7) * sr + P(8) * sg + P(9) * sb;
    
    error = error + sqrt((r - dr)^2 + (g - dg)^2 + (b - db)^2);
end  
end

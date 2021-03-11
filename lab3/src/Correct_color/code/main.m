clear;
clc;
close all;



A = load('src.txt');
B = load('dst.txt');

       

para0  = [  1;      0;      0;       0;       1;       0;       0;       0;      1 ];    
lb     = [  0.98;   -0.02;   -0.02;    -0.02;     0.98;     -0.02;    -0.02;    -0.02;   0.98 ];
ub     = [  1.02;      0.02;    0.02;     0.02;    1.02;      0.02;     0.02;    0.02;    1.02 ];


options = optimset('Display','iter-detailed','Algorithm','interior-point','FunValCheck','on',...
    'TolFun',10^-6,'LargeScale','off','TolX',10^-6,'MaxFunEvals',10^6,...
   'MaxIter',10000);

[respara, reserror, exitflag, output] = fmincon(@errorfunc, para0, [], [], [], [], lb, ub, [], options);

%options =  optimoptions(@fminunc,'Display','iter-detailed','Algorithm','quasi-newton','FunValCheck','on',...
%    'TolFun',10^-6,'TolX',10^-6,'MaxFunEvals',10^6,...
%   'MaxIter',10000);
%[respara, reserror, exitflag, output] = fminunc(@errorfunc, para0, options);
% disp(respara);
for i = 1 : size(respara, 1)
    if (i == size(respara, 1))
        disp(num2str(respara(i,1)));
    else 
        disp([num2str(respara(i,1)) ',']);
    end
end
% 
% 0.98,
% 0.002007,
% -0.02,
% -0.014843,
% 0.98,
% -0.02,
% -0.019999,
% -0.01737,
% 0.98

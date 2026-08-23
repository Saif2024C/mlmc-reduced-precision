%
% this MATLAB code evaluates the L_2 errors arising from
% different algorithms for converting (0,1) uniform random
% variables to approximate Normal random variables
%

close all
clear all


% method 1: piecewise constant LUT
%

ds    = 8:2:16;
errs1 = [];

for d = ds
  k  = 1:2^(d-1);
  du = 2^(-d);
  Z_ave  = norminv_int((k-1)*du,k*du)/du;
  Z2_ave = norminv2_int((k-1)*du,k*du)/du;

  err   = 2*du*sum(Z2_ave-Z_ave.^2);
  errs1 = [errs1 err];

  fprintf(1,'method 1: d = %2d, err = %f \n',d,err)
end

%
% method 2: dyadic
%

errs2   = [];
errs2_2 = [];

a = zeros(32);
b = zeros(32);
  
for pass = 1:2

  for d = ds
    du = 2^(-d);
    k  = 1:2^(d-1);
    Z_ave  = norminv_int((k-1)*du,k*du)/du;
    Z2_ave = norminv2_int((k-1)*du,k*du)/du;

    err = 0;

    m = floor(pass*log2(k-1)) + 1;
    m(1) = m(2)-1;

    for m2 = min(m):max(m)
      m3 = m2 - min(m) + 1;

      k2 = find(m==m2,1,'first'):find(m==m2,1,'last');
      n_sub = length(k2);

      if n_sub>1
        Z_ave2 = sum(Z_ave(k2))/n_sub;
        dn = (1:n_sub) - (n_sub+1)/2;
        dZ = Z_ave(k2) - Z_ave2;
        dZ = (dZ*dn') / (dn*dn');

        Z2(k2) = Z_ave2 + dn*dZ;
        err = err + 2*du*(sum(Z2_ave(k2)-Z_ave(k2).^2+(Z2(k2)-Z_ave(k2)).^2));
        a(m3) = dZ/du;
        b(m3) = Z_ave2 - dZ*(k2(1)+k2(end)-1)/2;
    
      elseif n_sub==1
        err = err + 2*du*(Z2_ave(k2)-Z_ave(k2).^2);
        a(m3) = 0;
        b(m3) = Z_ave(k2);
      else
        a(m3) = 0;
        b(m3) = 0;
      end
    end

    if pass==1
      errs2 = [errs2 err];
      fprintf(1,'method 2: d = %2d, err = %f \n',d,err)
    else
      errs2_2 = [errs2_2 err];
      fprintf(1,'method 2(b): d = %2d, err = %f \n',d,err)
    end
  end
end

%
% plot final results
%

figure; pos=get(gcf,'pos'); pos(3:4)=pos(3:4).*[0.8 0.8]; set(gcf,'pos',pos);

set(0,'DefaultAxesColorOrder',[0 0 0]);
set(0,'DefaultAxesLineStyleOrder','-.*|:*|--*|-*')

semilogy(ds,errs2, ds,errs2_2, ds,errs1)
xlabel('$d$ input bits','Interpreter','latex') 
ylabel('MSE') 
legend('dyadic', 'dyadic-2', 'LUT')
axis([8 16 1e-6 2e-3])

print('-deps2','fig1.eps')

%
% print out tabular information
%

fprintf(1,'\nbits       dyadic       dyadic-2     LUT \n')
for n = 1:length(ds)
  fprintf(1,' %2d   %12.7f  %12.7f  %12.7f \n', ...
	    ds(n), errs2(n), errs2_2(n), errs1(n))
end

%
% print out transformed tabular spline data;
% spline is used with y = 2^(-7) i and therefore
%
% x = 2^(-16)*(i+0.5) =  2^(-16)*(2^7*y + 0.5) = 2^(-9)*y + 2^(-17)
% ==> ax + b = 2^(-9)*a * y + (b + 2^(-17)*a)
	  
  b = b + 2^(-17)*a;
  a = 2^(-9)*a;

  fprintf(1,'\n Tabular data for piecewise linear spline \n\n');

  for m3 = 1:4:31
    fprintf(1,'%11.7ff16, %11.7ff16, %11.7ff16, %11.7ff16, \n',a(m3:m3+3));
  end

  fprintf(1,'\n');

  for m3 = 1:4:31
    fprintf(1,'%11.7ff16, %11.7ff16, %11.7ff16, %11.7ff16, \n',b(m3:m3+3));
  end


%
% this computes the integral of Phi^{-1}(u) over the interval [u1,u2]
%

function y = norminv_int(u1,u2)
  u1 = max(u1,1e-15);
  u2 = min(u2,1-1e-15);
  z1 = norminv(u1);
  z2 = norminv(u2);
  y  = normpdf(z1)-normpdf(z2);
end

%
% this computes the integral of (Phi^{-1}(u))^2 over the interval [u1,u2]
%

function y = norminv2_int(u1,u2)
  u1 = max(u1,1e-15);
  u2 = min(u2,1-1e-15);
  z1 = norminv(u1);
  z2 = norminv(u2);
  y  = z1.*normpdf(z1)-z2.*normpdf(z2)+(u2-u1);
end



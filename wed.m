

% M = 8/(1/(norm(Y(j)-1.5)^2+0.05) + 1/norm(X(i)-1)^2 + 1);
xmin=0;
xmax=8;
ymin=-4;
ymax=5;
dx=0.1;
[y_m,x_m] = meshgrid(ymin:dx:ymax,xmin:dx:xmax);
X=xmin:dx:xmax;
Y=ymin:dx:ymax;
Z = zeros(size(x_m,1),size(x_m,2));
ZZ = zeros(size(x_m,2),size(x_m,1));
for i=1:length(X)
    for j=1:length(Y)
        ZZ(j,i) = 8/(1/(norm(Y(j)-1.5)^2+0.05) + 1/norm(X(i)-1)^2 + 1);
        Z(i,j) = ZZ(end);
%         1/(1/norm([X(i);Y(j)]-[7;0])^2 + 1/norm([X(i);Y(j)]-[3;1])^2 + 1);
    end
end
figure(1)
surf(x_m,y_m,ZZ),shading flat
% surf(x_m,y_m,Z-max(max(Z))-0.5),shading flat
    xlabel('x(m)')
    ylabel('y(m)')
    grid on
%     axis([3,11,-2,5])
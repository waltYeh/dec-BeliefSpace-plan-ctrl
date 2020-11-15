xmin=0.5;
xmax=8;
ymin=-1;
ymax=5;
dx=0.1;
[x_m,y_m] = meshgrid(xmin:dx:xmax,ymin:dx:ymax);
X=xmin:dx:xmax;
Y=ymin:dx:ymax;
Z = zeros(size(x_m,1),size(x_m,2));
for i=1:length(X)
    for j=1:length(Y)
        ZZ = om.getObservationNoiseJacobian([0;0;X(i);Y(j)]);
        Z(j,i) = ZZ(end);
%         1/(1/norm([X(i);Y(j)]-[7;0])^2 + 1/norm([X(i);Y(j)]-[3;1])^2 + 1);
    end
end
figure(99)
surf(x_m,y_m,Z),shading flat
xlabel('x(m)')
ylabel('y(m)')
zlabel('M')
% title('Stärke des Messrauschens')
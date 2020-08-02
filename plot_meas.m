[x_m,y_m] = meshgrid(1:0.5:10,-1:0.5:5);
X=1:0.5:10;
Y=-1:0.5:5;
Z = zeros(size(x_m,1),size(x_m,2));
for i=1:length(X)
    for j=1:length(Y)
        Z(j,i) = 1/(1/norm([X(i);Y(j)]-[7;0])^2 + 1/norm([X(i);Y(j)]-[3;1])^2 + 1);
    end
end
figure(5)
surf(x_m,y_m,Z-1.2)
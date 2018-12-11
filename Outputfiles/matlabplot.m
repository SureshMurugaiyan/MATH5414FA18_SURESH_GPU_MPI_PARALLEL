clear all;
close all;
format long;

X = dlmread('xCellcenter.txt');
Y = dlmread('yCellcenter.txt');
U = dlmread('uxVelocity.txt');
V = dlmread('uyVelocity.txt');
[Nx,Ny] = size(X); % Grid size
Res = dlmread('ResidualPlotting.txt');
V1 = dlmread('uyHorzCenterVelocity_benchmark.txt');
V2 = dlmread('uxVertCenterVelocity_benchmark.txt');

x1 = linspace(-0.5,0.5,128);
y1 = linspace(-0.5,0.5,128);


figure (1)
colormap(parula(25))
contourf(X,Y,U,25)
title('Ux-Velocity')
xlabel('X-Coordinate');
ylabel('Y-Coordinate');
% saveas(gcf,'P1_Ux_Velocity.png')
saveas(gcf,'P1_Ux_Velocity','epsc')

figure  (2)
colormap(parula(25))
contourf(X,Y,V,25)
title('Uy-Velocity')
xlabel('X-Coordinate');
ylabel('Y-Coordinate');
% saveas(gcf,'P2_Uy_Velocity.png')
saveas(gcf,'P2_Uy_Velocity','epsc')


figure  (3)
% plot(X((Nx+1)/2,:),V((Nx+1)/2,:),'*')
plot(X((Nx)/2,:),V((Nx)/2,:),'*')
hold on
plot(x1',V1,'o'); % Ghia and GHia
legend('Current Work','Ghia & Ghia')
title('V-Velocity Horizontal Geometric Center')
xlabel('X-Coordinate');
ylabel('V-Velocity');
% saveas(gcf,'P3_V_Velocity_Horizontal_Geometric_Center.png')
saveas(gcf,'P3_V_Velocity_Horizontal_Geometric_Center','epsc')

figure  (4)
plot(Y(:,(Nx+1)/2),U(:,((Nx+1)/2)),'*')
hold on
plot(y1',V2,'o');
legend('Current Work','Ghia & Ghia')
title('U-Velocity Vertical Geometric Center')
xlabel('Y-Coordinate');
ylabel('U-Velocity');
% saveas(gcf,'P4_U_Velocity_Horizontal_Geometric_Center.png')
saveas(gcf,'P4_U_Velocity_Horizontal_Geometric_Center','epsc')


figure  (5)
semilogy(Res(:,1),Res(:,2))
title('L2 Norm of \Delta U')
xlabel('Iteration');
ylabel('L2 Norm of  \Delta U');
% saveas(gcf,'P5_L2_Norm_of_U.png')
saveas(gcf,'P5_L2_Norm_of_U','epsc')

figure  (6)
semilogy(Res(:,1),Res(:,3))
title('L2 Norm of  \Delta V')
xlabel('Iteration');
ylabel('L2 Norm of  \Delta V');
% saveas(gcf,'P6_L2_Norm_of_V.png')
saveas(gcf,'P6_L2_Norm_of_V','epsc')

figure  (7)
semilogy(Res(:,1),Res(:,4))
title('L2 Norm of  \Delta P')
xlabel('Iteration');
ylabel('L2 Norm of  \Delta P');
% saveas(gcf,'P7_L2_Norm_of_P.png')
saveas(gcf,'P7_L2_Norm_of_P','epsc')

figure  (8)
surface(X,Y,0*X)
axis image
title('GRID')
xlabel('X-Coordinate');
ylabel('Y-Coordinate');
hold on;
plot(X(1,:),Y(1,:),'r','Linewidth',2);
plot(X(Ny,:),Y(Ny,:),'r','Linewidth',2);
plot(X(:,1),Y(:,1),'r','Linewidth',2);
plot(X(:,Nx),Y(:,Nx),'r','Linewidth',2);
% saveas(gcf,'P8_Grid.png')
saveas(gcf,'P8_Grid','epsc')
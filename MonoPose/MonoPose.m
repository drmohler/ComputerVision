%David R Mohler 
%EE-5450: Topics in Robotics
%Project 1
%Spring 2018

clear
close all
%read in the images
i1=imread('Images\cvClass 023.jpg','jpg');
i2=imread('Images\cvClass 026.jpg','jpg');
i3=imread('Images\cvClass 024.jpg','jpg');
i4=imread('Images\cvClass 028.jpg','jpg');
lbox=45.6; %length of box (assume centimeters) 
hbox=32.5; %height of box
wbox=10.1; %width of box
n=7;

%use either the first set of commands (to initially mark correspondence
%points manually) or the load command  (to read in previous correspondence
%points)
%    X1=[1 1];X2=[1 1];%must start with an initial point; this will be removed later
%    [X1,X2]=cpselect(i1,i2,X1,X2,'Wait',true); 
%     save motorBoxCorners23_26 X1 X2 %save the correspondence points you just found

%load in correspondence points
load motorBoxCorners23_26.mat
load motorBoxCorners24_28.mat

[mc,nc]=size(X1); %need if changing # of Correspon. pts. 
X1=X1(2:n+1,:); %remove initial point, it is not good data
X2=X2(2:n+1,:);
[mc,nc]=size(X1);

x1pixmat=[X1'
    ones(1,mc)];  %convert the points to homogeneous coordinates
x2pixmat=[X2'
    ones(1,mc)];  %convert the points to homogeneous coordinates
x3pixmat=[X3'
    ones(1,mc)];  %convert the points to homogeneous coordinates
x4pixmat=[X4'
    ones(1,mc)];  %convert the points to homogeneous coordinates

%Corruption of data points with gaussian noise
x1pixcor = x1pixmat;
%corruption of each coordinate in each point seperately
% for i = 1:mc
%   x1pixcor(1,i) = x1pixmat(1,i)+10*randn(1,1);
%   x1pixcor(2,i) = x1pixmat(2,i)+10*randn(1,1);
%   x1pixcor(3,i) = x1pixmat(3,i)+10*randn(1,1);
% end

%Corrupting all coordinates equally across correspondence points
%(maintains shape, but is shifted in image plane) 
x1pixcor(1,:) = x1pixmat(1,:)+75*randn(1,1);
x1pixcor(2,:) = x1pixmat(2,:)+75*randn(1,1);
x1pixcor(3,:) = x1pixmat(3,:)+75*randn(1,1);

%enter object coordinates for the first 7 corners
Xomat=[0 lbox lbox lbox 0 0 lbox
       0 0 hbox hbox hbox 0 0
       0 0 0 -wbox -wbox -wbox -wbox]; %object coords of the four corners
Xoh = [Xomat; ones(1,mc)]; %Homogeneous object coordinates
%find calibration matrices
[gest1qr,lambda1qr,K1]=monoPoseQR(Xomat,x1pixmat); %find K, depth, and g
Rest1qr=gest1qr(1:3,1:3);Test1qr=gest1qr(1:3,4); %Extraction of R and T

[gest1cor,lambda1cor,K1cor]=monoPoseQR(Xomat,x1pixcor); %find K,depth, & g
Rest1cor=gest1cor(1:3,1:3);Test1cor=gest1cor(1:3,4); %Extraction of R and T

[gest2qr,lambda2qr,K2]=monoPoseQR(Xomat,x2pixmat);
Rest2qr=gest2qr(1:3,1:3);Test2qr=gest2qr(1:3,4);

[gest3qr,lambda3qr,K3]=monoPoseQR(Xomat,x3pixmat);
Rest3qr=gest3qr(1:3,1:3);Test3qr=gest3qr(1:3,4);

[gest4qr,lambda4qr,K4]=monoPoseQR(Xomat,x4pixmat);
Rest4qr=gest4qr(1:3,1:3);Test4qr=gest4qr(1:3,4);

PI1 = [K1*Rest1qr K1*Test1qr];
PI1cor = [K1cor*Rest1cor K1cor*Test1cor];
PI2 = [K2*Rest2qr K2*Test2qr];
PI3 = [K3*Rest3qr K3*Test3qr];
PI4 = [K4*Rest4qr K4*Test4qr];

%Estimated pixel coordinates of correspondence points
x1pixest = zeros(size(x1pixmat));
x1pixestcor = zeros(size(x1pixmat));
x2pixest = zeros(size(x2pixmat));
x3pixest = zeros(size(x3pixmat));
x4pixest = zeros(size(x4pixmat));
for i = 1:mc
    x1pixest(:,i) = (PI1*Xoh(:,i))/lambda1qr(i);
    x1pixestcor(:,i) = (PI1cor*Xoh(:,i))/lambda1cor(i);
    x2pixest(:,i) = (PI2*Xoh(:,i))/lambda2qr(i);
    x3pixest(:,i) = (PI3*Xoh(:,i))/lambda3qr(i);
    x4pixest(:,i) = (PI4*Xoh(:,i))/lambda4qr(i);
end 

%Reconstruction of Object Coordinates from estimates
Xoest1 = zeros(size(Xomat));
Xoest1cor = zeros(size(Xomat));
Xoest2 = zeros(size(Xomat));
Xoest3 = zeros(size(Xomat));
Xoest4 = zeros(size(Xomat));
for i = 1:mc
   Xoest1(:,i) = Rest1qr'*inv(K1)*(lambda1qr(i)*x1pixest(:,i)-K1*Test1qr);
   Xoest1cor(:,i) = Rest1cor'*inv(K1cor)*...
                        (lambda1cor(i)*x1pixestcor(:,i)-K1cor*Test1cor);
   Xoest2(:,i) = Rest2qr'*inv(K2)*(lambda2qr(i)*x2pixest(:,i)-K2*Test2qr); 
   Xoest3(:,i) = Rest3qr'*inv(K3)*(lambda3qr(i)*x3pixest(:,i)-K3*Test3qr); 
   Xoest4(:,i) = Rest4qr'*inv(K4)*(lambda4qr(i)*x4pixest(:,i)-K4*Test4qr); 
end

%Average error in each direction and total distance
avXRMSE = (sqrt(sum(((Xomat(1,:)-Xoest1(1,:)).^2)/mc))+...
    sqrt(sum(((Xomat(1,:)-Xoest2(1,:)).^2)/mc))+...
    sqrt(sum(((Xomat(1,:)-Xoest3(1,:)).^2)/mc))+...
    sqrt(sum(((Xomat(1,:)-Xoest4(1,:)).^2)/mc)))/4;
avYRMSE = (sqrt(sum(((Xomat(1,:)-Xoest1(1,:)).^2)/mc))+...
    sqrt(sum(((Xomat(2,:)-Xoest2(2,:)).^2)/mc))+...
    sqrt(sum(((Xomat(2,:)-Xoest3(2,:)).^2)/mc))+...
    sqrt(sum(((Xomat(2,:)-Xoest4(2,:)).^2)/mc)))/4;
avZRMSE = (sqrt(sum(((Xomat(1,:)-Xoest1(1,:)).^2)/mc))+...
    sqrt(sum(((Xomat(3,:)-Xoest2(3,:)).^2)/mc))+...
    sqrt(sum(((Xomat(3,:)-Xoest3(3,:)).^2)/mc))+...
    sqrt(sum(((Xomat(3,:)-Xoest4(3,:)).^2)/mc)))/4;
%DO AVERAGE DISTANCE PRMSE
% distx = sqrt((Xomat(1,:)-Xoest1(1,:)).^2+(Xomat(2,:)-Xoest1(2,:)).^2)
% avDRMSE = (sqrt(sum(distx/mc))+...
%     sqrt(sum(((Xomat(3,:)-Xoest2(3,:)).^2)/mc))+...
%     sqrt(sum(((Xomat(3,:)-Xoest3(3,:)).^2)/mc))+...
%     sqrt(sum(((Xomat(3,:)-Xoest4(3,:)).^2)/mc)))/4;

%Images and plotting 
figure(1)
clf
image(i1);
hold on
plot(X1(:,1)+j*X1(:,2),'g*')
scatter(x1pixest(1,:),x1pixest(2,:),'r*') %Pure pixel estimates
scatter(x1pixestcor(1,:),x1pixestcor(2,:),'b^') %Noise corrupted pixel est.
legend('Image Points','Estimated Points','Corrupted estimate')
title('Image 1')
hold off

figure(2)
clf
image(i2)
hold on
plot(X2(:,1)+j*X2(:,2),'g*')
scatter(x2pixest(1,:),x2pixest(2,:),'r*')
legend('Image Points','Estimated Points')
title('Image 2')
hold off


figure(3)
clf
image(i3)
hold on
plot(X3(:,1)+j*X3(:,2),'g*')
scatter(x3pixest(1,:),x3pixest(2,:),'r*')
legend('Image Points','Estimated Points')
title('Image 3')
hold off

figure(4)
clf
image(i4)
hold on
plot(X4(:,1)+j*X4(:,2),'g*')
scatter(x4pixest(1,:),x4pixest(2,:),'r*')
legend('Image Points','Estimated Points')
title('Image 4')
hold off

figure(5)
hold on
clf
pts=1:7;
plot3(Xomat(1,pts),Xomat(2,pts),Xomat(3,pts),'--rh')
axis equal
title('3D Box Corners')
hold off

figure(6)
hold on
clf
pts=1:7;
plot3(Xoest1(1,pts),Xoest1(2,pts),Xoest1(3,pts),'--g*')
axis equal
title('Reconstructed 3D Box Corners')
hold off

%Reconstruction of truth and noise boxes
figure(7)
clf
pts=1:7;
hold on
plot3(Xomat(1,pts),Xomat(2,pts),Xomat(3,pts),'--rh')
plot3(Xoest1cor(1,pts),Xoest1cor(2,pts),Xoest1cor(3,pts),'--b^')
view(3)
axis equal
title('Reconstructed 3D Box Corners (Noisy)')
legend('Ground Truth','Reconstructed Points')
hold off

figure(8)
clf
pts=1:7;
hold on
plot3(Xomat(1,pts),Xomat(2,pts),Xomat(3,pts),'--rh')
plot3(Xoest1(1,pts),Xoest1(2,pts),Xoest1(3,pts),'--k^')
view(3)
axis equal
title('Reconstructed 3D Box Corners (Noisy)')
legend('Ground Truth','Reconstructed Points')
hold off

figure(1)

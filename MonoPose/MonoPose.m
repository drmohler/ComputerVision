%David R Mohler 
%EE-5450: Topics in Robotics
%Project 1
%Spring 2018

clear
%read in the images
i1=imread('Images\cvClass 023.jpg','jpg');
i2=imread('Images\cvClass 026.jpg','jpg');
lbox=45.6; %length of box
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


[mc,nc]=size(X1);
X1=X1(2:n+1,:); %remove initial point, it is not good data
X2=X2(2:n+1,:);
[mc,nc]=size(X1);

x1pixmat=[X1'
    ones(1,mc)];  %convert the points to homogeneous coordinates
x2pixmat=[X2'
    ones(1,mc)];  %convert the points to homogeneous coordinates



%enter object coordinates for the first 7 corners
Xomat=[0 lbox lbox lbox 0 0 lbox
       0 0 hbox hbox hbox 0 0
       0 0 0 -wbox -wbox -wbox -wbox]; %object coordinates of the four corners

%find calibration matrices
% [gest1qr,lambda1qr,K1]=monoPoseQR(Xomat,x1pixmat);%write your own function here
% Rest1qr=gest1qr(1:3,1:3);Test1qr=gest1qr(1:3,4);
% 
% [gest2qr,lambda2qr,K2]=monoPoseQR(Xomat,x2pixmat);%write your own function here
% Rest2qr=gest2qr(1:3,1:3);Test2qr=gest2qr(1:3,4);


%find calibrated image positions, pose, 3d world positions
%
%the code below may be used, with your modifications, to help with
%plotting.  


figure(21)
clf
image(i1);
hold on
plot(X1(:,1)+j*X1(:,2),'g*')
legend('Image Points')
title('Image 1')
hold off

figure(22)
clf
image(i2)
hold on
plot(X2(:,1)+j*X2(:,2),'g*')
legend('Image Points')
title('Image 2')
hold off

figure(23)
clf
pts=1:7;
plot3(Xomat(1,pts),Xomat(2,pts),Xomat(3,pts),'--rh')
axis equal
title('3D Box Corners')



function [g,lambda,K]=monoPoseQR(Xomat,xpixh)
%Function to find the camera pose (g), depth to object points (lambda), and 
%camera calibration matrix (K)

n = size(Xomat,2); 

%convert object coords to homogeneous coordinates
Xoh = [Xomat; ones(1,n)];
I3 = eye(3);
e3 = [0 0 1]';

N = zeros(3*n,12);
j = 1;
for i  = 1:n
    N(j:j+2,:) = kron(Xoh(:,i)',I3)-kron(Xoh(:,i)',(xpixh(:,i)*e3'));
    j = j+3; 
end

[~,~,V] = svd(N);
PIs = V(:,end); %Extract LSE of stacked PI
PI = reshape(PIs,[3 4]); %Unstack PI vector
[alf,K,R] = qrCommute(PI(:,1:3)); %use QR decomp to find K,R, and scale 
T = (inv(K)/alf)*PI(:,4); %Calculate translation
PIest = [K*R K*T];
g = [R T;0 0 0 1]; %Contruct pose matrix 
lambda = e3'*PIest*Xoh; %Extract depth

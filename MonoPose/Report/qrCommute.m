function [alf, K,R ] = qrCommute( M )
%function [alf, K,R ] = qrCommute( M )

%modified qr decomposition.  It commutes the order
%of the rotation matrix and the upper triangular matrix.
% 
%The decomposition is returned such that alf*K*R = M, where
%alf is a scalar, K is upper triangular, and R is a special orthogonal 
%matrix.  
%
%In addition, it makes the diagonal of K positive, with the last entry 
%of K, K(n,n), equal to one.  This factorization is ideal for uncalibrated 
%camera factorization problems.  If it is not possible, an error is
%returned.
%
%J. McInroy, 2/4/10

[m,n]=size(M);
E=zeros(n,n);
E2=zeros(n,n);
for i=1:n,
    E(i,n+1-i)=1;
end
[Q,Rp]=qr(M'*E);
Kn=E*Rp';
Rn=Q';
alf=Kn(n,1);
for i=1:n,
    if Kn(i,n+1-i)<0,
        E2(n+1-i,i)=-1;
    else
       E2(n+1-i,i)=1;        
    end
end
R=E2'*Rn;
K=Kn*E2/alf;
if det(R)<0,
    R=-R;
    alf=-alf;
end
if K(n,n)<0,
    K=-K;
    alf=-alf;
end
end


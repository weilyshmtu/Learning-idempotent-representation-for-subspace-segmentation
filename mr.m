function [M,Z] = mr(X,initial_alg,lambda1,lambda2,beta)


[d,n] = size(X);
switch initial_alg
    case 'SSC'
        W = ssc(X,lambda1);
    case 'LRR'
        W = lrr(X,lambda1);
end   

 M = funM1(W,n,lambda2);
 Z = funZ1(M,n,beta);

%% Searching M and Z by using classical ALM
function M1 = funM1(W,n,lambda2)
% coded by using the classical ALM 
disp(['M is updating...'])
iter = 0;
maxiter = 1e4;
mu = 1e-6;
maxmu = 1e30;
rho = 1.1;
tol = 1e-6;
I = eye(n);

M1 = zeros(n,n);
M2 = zeros(n,n);
Y = zeros(n,n);
N = ones(n,n); 

while iter < maxiter
    iter = iter + 1;
    
    %% update M1
    M1 = (mu*(N - M2) - Y)/(2*lambda2 + mu);
    M1 = max(M1,0);
    M1 = M1 - diag(diag(M1)) + I;
    %% update M2
    temp = N - M1 - Y/mu;
    M2 = max(temp - W/mu,0) + min(temp + W/mu,0);
    M2 = max(M2,0);

    stopC = max(max(abs(M1 + M2 -N)));
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',stopC =' num2str(stopC,'%2.3e')]);
    end
    if stopC < tol
        disp(['M is updated!']);
        disp('--------------------')
        break;
    else
        Y = Y + mu*(M1 + M2 - N);
        mu = min(maxmu,mu*rho);
    end
    
end

function Z1 = funZ1(M,n,beta)
% coded by using the classical ALM 
disp(['Z is updating...'])
iter = 0;
maxiter = 3e3;
mu = 1e-6;
maxmu = 1e30;
rho = 1.1;
tol = 1e-3;
e = ones(1,n);
ete = e'*e;
I = eye(n);
N = ones(n,n);
H = N - M;
c = norm(H,1)*beta/n;
alpha = 0:-0.01:-5;

Z2 = N;
Y1 = zeros(1,n);
Y2 = zeros(n,n);
while iter < maxiter
    iter = iter + 1;
    %% update Z1
    A = mu*(ete + I);
    B = mu*(ete + Z2) - e'*Y1 - Y2 - I;
    Z1 = A\B;
    
    %% update Z2
    T = Z1 + Y2/mu;
    T = max(T,0);
    if norm(H.*T,1) > c
         for j=1:length(alpha)
            TN = max(T + alpha(j)*H,0);
            if norm(H.*TN,1)<=c
                Z2 = max(T + alpha(j)*H,0); 
                 if iter==1 || mod(iter,100)==0
                    disp('Z2 is updated!')
                 end
                break;
            end
        end
    else
        Z2 = T;
        if iter==1 || mod(iter,100)==0
               disp('Z2 is updated as T!')
        end
    end
    
     stopC = max([max(max(abs(e*Z1-e))),max(max(abs(Z1-Z2)))]);
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',stopC =' num2str(stopC,'%2.3e')]);
    end
    if stopC < tol
        disp('Z is updated!');
        disp('--------------------')
        break;
      
    else
        Y1 = Y1 + mu*(e*Z1 - e);
        Y2 = Y2 + mu*(Z1 - Z2);
        mu = min(maxmu,mu*rho);
    end
    
end


%% Searching M and Z by using methods proposed in the corresponding reference
function M1 = funM2(W,n,lambda2)
% coded by following the method proposed in 
% M. Lee, J. Lee, H. Lee, N. Kwak, Membership representation for detecting block-diagonal structure in low-rank or sparse subspace clustering,  CVPR2015.
disp(['M is updating...'])
iter = 0;
maxiter = 1e4;
mu = 1e-6;
maxmu = 1e30;
rho = 1.1;
tol = 1e-6;

M1 = zeros(n,n);
M2 = zeros(n,n);
Y = zeros(n,n);
e = ones(1,n);
B = 1/(2*lambda2)*sign(W).*W;
while iter < maxiter
    iter = iter + 1;
    %% update M1
    M1 = max(mu*M2 + B - Y,0)/(1+mu);
     M1 = M1 - diag(diag(M1)) + diag(e);
    %% update M2
    T = M1 + Y/mu;
    T = (T + T')/2;
    [U,S,V] = svd(T);
    M2 = U*max(S,0)*V';
    
    stopC = max(max(abs(M1-M2)));
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',stopC =' num2str(stopC,'%2.3e')]);
    end
    if stopC < tol
        disp(['M is updated!']);
        disp('--------------------')
        break;
    else
        Y = Y + mu*(M1 - M2);
        mu = min(maxmu,mu*rho);
    end
    
end

function Z1 = funZ2(M,n,beta)
% coded by following the method proposed in 
% M. Lee, J. Lee, H. Lee, N. Kwak, Membership representation for detecting block-diagonal structure in low-rank or sparse subspace clustering,  CVPR2015.
disp(['Z is updating...'])
iter = 0;
maxiter = 1e4;
mu = 1e-6;
maxmu = 1e30;
rho = 1.1;
tol = 1e-3;

I = eye(n);
e = ones(1,n);
ete = e'*e;
H = e'*e/n - M ; % a mistake in the reference 
c = norm(H,1)*beta/n;
alpha = 0:-0.01:-5;

Z2 = M;
Y = zeros(n,n);

while iter < maxiter
    iter = iter + 1;
    
    %% update Z1
    tI = I - 1/n*ete;
    T = tI*(Z2 - (I + Y)/mu)*tI +  1/n*ete;
    T = (T + T')/2;
    [U,S,V] = svd(T);
    Z1 = U*max(S,0)*V';
    
    %% update Z2
    T = Z1 + Y/mu;
    T = max(T,0);
    if norm(H.*T,1) > c
         for j=1:length(alpha)
            TN = max(T + alpha(j)*H,0);
            if norm(H.*TN,1)<=c
                Z2 = max(T + alpha(j)*H,0); 
                 if iter==1 || mod(iter,100)==0
                    disp('Z2 is updated!')
                 end
                break;
            end
        end
    else
        Z2 = T;
        if iter==1 || mod(iter,100)==0
               disp('Z2 is updated as T!')
        end
    end
    
    stopC = max(max(abs(Z1-Z2)));
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',stopC =' num2str(stopC,'%2.3e')]);
    end
    if stopC < tol
        disp('Z is updated!');
        disp('--------------------')
        break;
      
    else
        Y = Y + mu*(Z1 - Z2);
        mu = min(maxmu,mu*rho);
    end
    
end

%% Algorithms for obtaining the initial coefficient matrices
function [Z] = ssc(X,lambda)
%This routine solves the following l1-norm 
% optimization problem with l1-error
% min |Z|_1+lambda*|E|_1
% s.t., X = XZ+E
%       Zii = 0 (i.e., a data vector can not rerepsent itselft)
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
if nargin<2
    lambda = 1;
end
tol = 1e-7;
maxIter = 1e6;
[d n] = size(X);
rho = 1.1;
max_mu = 1e30;
mu = 1e-3;
xtx = X'*X;
inv_x = inv(xtx+eye(n));
%% Initializing optimization variables
% intialize
J = zeros(n);
E = sparse(d,n);
Z = J;

Y1 = zeros(d,n);
Y2 = zeros(n);
%% Start main loop
iter = 0;
while iter<maxIter
    iter = iter + 1;
    
    temp = Z + Y2/mu;
    J = max(0,temp - 1/mu)+min(0,temp + 1/mu);
    J = J - diag(diag(J)); %Jii = 0
    
    Z = inv_x*(xtx-X'*E+J+(X'*Y1-Y2)/mu);
    
    xmaz = X-X*Z;
    temp = X-X*Z+Y1/mu;
    E = mu*temp/(2*lambda+mu);
     
    leq1 = xmaz-E;
    leq2 = Z-J;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end

function [Z,E] = lrr(X,lambda)
% This routine solves the following nuclear-norm optimization problem 
% by using inexact Augmented Lagrange Multiplier, which has been also presented 
% in the paper entitled "Robust Subspace Segmentation 
% by Low-Rank Representation".
%------------------------------
% min |Z|_*+lambda*|E|_2,1
% s.t., X = XZ+E
%--------------------------------
% inputs:
%        X -- D*N data matrix, D is the data dimension, and N is the number
%             of data vectors.
if nargin<2
    lambda = 1;
end
tol = 1e-6;
maxIter = 1e6;
[d n] = size(X);
rho = 1.1;
max_mu = 1e30;
mu = 1e-2;
xtx = X'*X;
inv_x = inv(xtx+eye(n));
%% Initializing optimization variables
% intialize
J = zeros(n,n);
Z = zeros(n,n);
E = sparse(d,n);

Y1 = zeros(d,n);
Y2 = zeros(n,n);
%% Start main loop
iter = 0;
disp(['initial,rank=' num2str(rank(Z))]);
while iter<maxIter
    iter = iter + 1;
    
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
 %   [U,sigma,V] = lansvd(temp,30,'L');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    
    Z = inv_x*(xtx-X'*E+J+(X'*Y1-Y2)/mu);
    
    xmaz = X-X*Z;
    temp = X-X*Z+Y1/mu;
    E = solve_l1l2(temp,lambda/mu);
    %E = max(0,temp - lambda/mu)+min(0,temp + lambda/mu);

    leq1 = xmaz-E;
    leq2 = Z-J;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        mu = min(max_mu,mu*rho);
    end
end

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end


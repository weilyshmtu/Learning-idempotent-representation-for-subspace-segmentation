function [Z,S,E] = idr(X,k,gamma,lambda)
% min||Z - S||_F^2 + gamma ||S - S^2|| + lambda ||E||_F^2
% X = XZ + E, eS = e, S>0, S = S',tr(S)=k;
%% parameters
tol = 1e-7;
maxIter = 1e4;
rho = 1.1;
max_mu = 1e30;
mu = 1e-6;

warning off
%% Initializing optimization variables
[d,n] = size(X);
e = ones(1,n);

C = ones(n,n);
Z = ones(n,n);
S = zeros(n,n);
D = zeros(n,n);
E = zeros(d,n);
I = eye(n);
XtX= X'*X ;
ete = e'*e;

Y1 = zeros(d,n);
Y2 = zeros(n,n);
Y3 = zeros(1,n);
Y4 = zeros(n,n);
%% Start main loop
iter  = 0;
while iter < maxIter
    iter = iter + 1;
    
    %% update Z
    A = 2*I + mu*XtX;
    B = 2*S + mu*(XtX - X'*E) + X'*Y1;
    
    Z = A\B;
    if sum(isnan(Z))>0
        Z = pinv(A)*B;
    end

    %% update S
    I_C = I - C;
    A = 2*(1+mu)*I + 2*gamma*(I_C*I_C');
    B = 2*Z + mu*C - Y2 + mu*D  - Y4;
    S = B/A;
    if sum(isnan(S))>0
        S = B*pinv(A);
    end

    %% update C
    A = 2*gamma*(S'*S) + mu*(I + ete);
    B = 2*gamma*(S'*S) + mu*S + Y2 + mu*ete - e'*Y3;

    C = A\B;
     if sum(isnan(C))>0
        C = pinv(A)*B;
    end

    C = max(C,0);
    C = 0.5*(C + C');
    
    
    %% update D
   
    A = S + Y4/mu;
    t = diag(A);    
    D = A - diag(t);
    tau = 1;
    eta = 2*(tau*k - sum(t))/n;
    d = t + eta/2;
%     d = quadprog(I,-t',[],[],e,k,[],[],[],opts);
    D = D + diag(d);
    %% update E
    A = mu*(X - X*Z) + Y1;
    E = A/(2*lambda +mu);
%     E = solve_l1l2(A/mu,beta/mu);


    leq1 = X - X*Z - E;
    leq2 = S - C;
    leq3 = e*C - e;
    leq4 = S - D;

    stopC = max([max(max(abs(leq1))),max(max(abs(leq2))),max(max(abs(leq3))),max(max(abs(leq4)))]);

    
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(Z,1e-3*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        Y4 = Y4 + mu*leq4;
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

function [A] = dssc(X,lambda,eta,gamma,type)
% min_{C,A} 1/2||X - XC||_F^2 + eta1/2|||Z|-eta2*A||_F^2 +eta3||Z||_1
% A is a doubly stochastic matrix,diag(Z)=0
% eta1 == lambda, eta2 == eta, eta3 == gamma 

[d,n] = size(X);
iter = 0;
t = 0;
maxiter = 1e3;
mu = 1e-6;
maxmu = 1e30;
rho = 1.1;
tol = 1e-4;

e = ones(n,1);
eet = e*e';
I = eye(n);
ttol = 1e-2;
switch type
    case 'JDSSC'
        tau = 0.001;
        Cp = zeros(n,n);
        Cn = zeros(n,n);
        A = zeros(n,n);
        Y = A;
        Z = X*(Cp-Cn);
        
        l1 = zeros(n,1);
        l2 = zeros(n,1);
        L1 = zeros(n,n);
        L2 = zeros(d,n);
        
        while iter < maxiter
            iter = iter + 1;
            
            %% update Cp,Cn
            while t < maxiter
                t = t + 1;
                Cpt = Cp - tau*(-X'*L2 + mu*X'*(X*(Cp-Cn)-Z));
                Cp = 1/(lambda + 1/tau)*(1/tau*Cpt - lambda*Cn + lambda*eta*A -gamma*eet);
                Cp = max(Cp,0);
                Cp = Cp - diag(diag(Cp));
                
                Cnt = Cn - tau*(X'*L2 - mu*X'*(X*(Cp-Cn)-Z));
                Cn = 1/(lambda + 1/tau)*(1/tau*Cnt - lambda*Cp + lambda*eta*A -gamma*eet);
                Cn = max(Cn,0);
                Cn = Cn - diag(diag(Cn));
                if norm(Cp-Cpt,'inf') <= ttol && norm(Cn-Cnt,'inf')<= ttol
%                     disp('Cp and Cn are updated!')
                    break;
                end
               
            end
           
            %% update A
            A = max(1/(lambda*eta^2 + mu)*(lambda*eta*(Cp + Cn) + L1 + mu*Y),0);
            
            %% update Y
            P = I - 1/(2*n + 1)*eet;
            V = mu*A + 2*mu*eet - e*l1' - l2*e' - L1;
            Y = 1/mu*(V - 1/(n+1)*P*V*eet - 1/(n+1)*eet*V*P);
        
            %% update Z
            Z = 1/(1+mu)*(X - L2 + mu*X*(Cp - Cn));
        
        
            leq1 = Y'*e - e;
            leq2 = Y*e - e;
            leq3 = Y - A;
            leq4 = Z - X*(Cp - Cn);
            
            stopC = max([max(max(abs(leq1))),max(max(abs(leq2))),max(max(abs(leq3))),max(max(abs(leq4)))]);
            if iter==1 || mod(iter,50)==0 || stopC<tol
                disp(['iter ' num2str(iter) ',stopC =' num2str(stopC,'%2.3e')]);
            end
            if stopC < tol
                break;
            else
                l1 = l1 + mu*leq1;
                l2 = l2 + mu*leq2;
                L1 = L1 + mu*leq3;
                L2 = L2 + mu*leq4;
                mu = min(maxmu,mu*rho);
            end   
        end
        
     case 'ADSSC'
         
         Z = elasticnet(X,lambda,gamma);
         absZ = abs(Z);
         
         % searching for A by column and row wise which is more efficient 
         % than the method mentioned in the corresponding reference  
         Ar = zeros(n,n);
         for i=1:n
            Ar(i,:) = 1/n*(1 - (sum(absZ(i,:))/eta))*e' + absZ(i,:)/eta;
            Ar(i,:) = max(Ar(i,:),0);
         end
         Ac = zeros(n,n);
         for j=1:n
            Ac(:,j) = 1/n*(1 - (sum(absZ(:,j))/eta))*e + absZ(:,j)/eta;
            Ac(:,j) = max(Ac(:,j),0);
         end
         A = (Ac + Ar)/2;
end



%% Elastic Net, an algorithm to obtain the initial coefficient matrix for ADSSC
function [Z] = elasticnet(X,eta1,eta3)

% 1/2\|X-XZ\|_F^2 + lambda1/2\|Z\|_F^2 + lambda2\|Z\|_1
% diag(Z)=0
[~,n] = size(X);

iter = 0;
maxiter = 1e4;
mu = 1e-6;
maxmu = 1e30;
rho = 1.1;
tol = 1e-6;


I = eye(n);
XtX = X'*X;

C = zeros(n,n);
Y = zeros(n,n);
while iter < maxiter
    iter = iter + 1;
    %% update Z
    A = XtX + (eta1 + mu)*I;
    B = XtX + mu*C - Y;
    Z = A\B;
    
    %% update C
    temp = Z + Y/mu;
    C = max(0,temp - eta3/mu) + min(0,temp + eta3/mu);
    
    leq1 = Z - C;
    stopC = max(max(abs(leq1)));
    if iter==1 || mod(iter,50)==0 || stopC<tol
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        Y = Y + mu*leq1;
        mu = min(maxmu,mu*rho);
    end
    
end




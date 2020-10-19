function sparse_feedback(F_in,diff,b_nom,options)
F=F_in;
rho=options.rho;
gam=options.gam;
% reweighted_Max_Iter = options.reweightedIter;
n_b=size(F,2);
n_u=size(F,1);
horizon=size(F,3);
n_blk=4;
G=F;

lam_sparse=zeros(n_u,n_b,horizon);
W=ones(n_u,n_b);%no dim of horizon, weight is identical along the horizon
W_blk=ones(n_blk);
% absolute and relative tolerances for the stopping criterion of ADMM
eps_abs = 1.e-4;
eps_rel = 1.e-2;

% stopping criterion tolerance for the Anderson-Moore method and Newton's
% method
tolAM = 1.e-2; 
tolNT = 1.e-3;
reweighted_Max_Iter =3;
ADMM_Max_Iter = 10;
for reweightedstep = 1 : reweighted_Max_Iter
    % Solve the minimization problem using ADMM
    for ADMMstep = 1 : ADMM_Max_Iter
        U=G-lam_sparse/rho;
        F=fmin_admm(diff,rho,F,U,b_nom,tolAM);
        V=F+lam_sparse/rho;
        sub_mat_amount=[n_u,n_b];
        blksize=[1;1];
        G=block_schr(V,W,gam,rho,blksize,sub_mat_amount);
        lam_sparse=lam_sparse+rho*(F-G);
        % nn is the number of states of each subsystem
        % mm is the number of inputs of each subsystem
        % N is the number of subsystems                
        Wnew = ones(sub_mat_amount);
        mm = blksize(1);
        nn = blksize(2);            
        eps = 1e-2;
        for ii = 1:sub_mat_amount(1)
            for jj = 1:sub_mat_amount(2)
                Wnew(ii,jj) = 1 / ( norm( F( mm*(ii-1)+1 : mm*ii, ...
                    nn*(jj-1)+1 : nn*jj ),'fro' ) + eps );
            end
        end 
        W=Wnew;
    end
end

end
function Fnew=fmin_admm(diffs,rho,F,U,b_nom,tol)
cuu=diffs.cuu;
cbb=diffs.cxx;
n_b=size(F,2);
n_u=size(F,1);
horizon=size(F,3)+1;
F_diff=zeros(size(F));
grad_phi=zeros(size(F));
cov_belief=zeros(n_b,n_b,horizon);
Ffullstep=zeros(size(F));
Fnew=zeros(size(F));
phi=zeros(horizon-1,1);
AM_Max_Iter = 100;
for k=1:horizon-1
    components_amount=2;
    [x, P, w] = b2xPw(b_nom(1:42,k), 4, components_amount);
    for j_comp=1:components_amount
        ind_states=(j_comp-1)*21+1:(j_comp-1)*21+4;
        cov_belief(ind_states,ind_states,k)=P{j_comp};
    end
    for j_assist=1:3
        ind_b=43+(j_assist-1)*6:42+(j_assist)*6;
        ind_b_start=ind_b(1);
        [x_ass, P_ass] = b2xP(b_nom(ind_b,k), 2);
        cov_belief(ind_b_start:ind_b_start+1,ind_b_start:ind_b_start+1,k)=P_ass;
    end
    phi(k) = trace( cov_belief(:,:,k) * (cbb(:,:,k) + F(:,:,k)' * cuu(:,:,k) * F(:,:,k)) ) + ...
                    (rho/2) * norm( F(:,:,k) - U(:,:,k), 'fro' )^2;
    
end
for iter=1:1%AM_Max_Iter
    for k=1:horizon-1
        Ffullstep(:,:,k)=lyap(rho*inv(cuu(:,:,k)),cov_belief(:,:,k),-rho*cuu(:,:,k)\U(:,:,k));
        F_diff(:,:,k)=Ffullstep(:,:,k)-F(:,:,k);
        grad_phi(:,:,k)=cuu(:,:,k)*F(:,:,k)*cov_belief(:,:,k) + rho * (F(:,:,k) - U(:,:,k));
        if trace( F_diff(:,:,k)' * grad_phi(:,:,k) ) > 1.e-10
            error('Ftilde is not a descent direction!')
        end 
        if norm( grad_phi(:,:,k), 'fro' ) < tol%found necessary condition for opt, above (NC-F)
%             break;
        end
        stepsize = 1;%step size will be a half for each while iteration
        while 1
            Ftemp = F(:,:,k) + stepsize * F_diff(:,:,k);
            phitemp_k = trace( cov_belief(:,:,k) * (cbb(:,:,k) + Ftemp' * cuu(:,:,k) * Ftemp) ) + ...
                            (rho/2) * norm( Ftemp - U(:,:,k), 'fro' )^2;
            alpha = 0.3; 
            beta  = 0.5;
            if ~isnan(phitemp_k) && phi(k) - phitemp_k > ...
                        stepsize * alpha * trace( - F_diff(:,:,k)' * grad_phi(:,:,k) )
%                 stepsize
%                 k
                break;
            end                
            stepsize = stepsize * beta;

            if stepsize < 1.e-16            
                error('Extremely small stepsize in F-minimization step!');            
            end 
        end%while
        Fnew(:,:,k) = Ftemp;    
    % 	L = Ltemp;
        phi(k) = phitemp_k;
    end%k
end%iter
end
function G=block_schr(V,W,gam,rho,blksize,sub_mat_amount)
G=zeros(size(V));
horizon=size(G,3)+1;
mm=blksize(1);
nn=blksize(2);
p = sub_mat_amount(1);
q = sub_mat_amount(2);
for k=1:horizon-1
    for i=1:p
        for j=1:q
            wij = W(i,j);
            Vij = V( mm * (i-1) + 1:mm*i, nn * (j-1) + 1:nn*j);
            a = (gam / rho) * wij;
            nVij = norm( Vij, 'fro' );
            if nVij <= a
                G( mm * (i-1) + 1:mm*i, nn * (j-1) + 1:nn*j,k) = 0;
            else                 
                G( mm * (i-1) + 1:mm*i, nn * (j-1) + 1:nn*j,k) = ...
                    (1 - a/nVij) * Vij;%(14)
            end
        end
    end
end
end
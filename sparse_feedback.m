function sparse_feedback(F,diff,b_nom,options)
rho=options.rho;
gam=options.gam;
reweighted_Max_Iter = options.reweightedIter;
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
ADMM_Max_Iter = 50;
for reweightedstep = 1 : reweighted_Max_Iter
    % Solve the minimization problem using ADMM
    for ADMMstep = 1 : ADMM_Max_Iter
        U=G-lam_sparse/rho;
        F=fmin_admm(diff,rho,F,U,b_nom,tolAM);
        V=F+lam_sparse/rho;
        G=block_schr(V,W,gam,rho,blksize,sub_mat_size);
        lam_sparse=lam_sparse+rho*(F-G);
        % nn is the number of states of each subsystem
        % mm is the number of inputs of each subsystem
        % N is the number of subsystems                
        Wnew = ones(sub_mat_size);
        mm = blksize(1);
        nn = blksize(2);            
        eps = 1e-3;
        for ii = 1:sub_mat_size(1)
            for jj = 1:sub_mat_size(2)
                Wnew(ii,jj) = 1 / ( norm( F( mm*(ii-1)+1 : mm*ii, ...
                    nn*(jj-1)+1 : nn*jj ),'fro' ) + eps );
            end
        end 
    end
end

end
function Fnew=fmin_admm(diffs,rho,F,U,b_nom,tolAM)
cuu=diffs.cuu;
n_b=size(F,2);
n_u=size(F,1);
horizon=size(F,3)+1;
nabla_JF=zeros(size(F));
cov_belief=zeros(n_b,n_b,horizon);
Fnew=zeros(size(F));
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
%     nabla_JF(:,:,k)=cuu(:,:,k)*F(:,:,k)*cov_belief(:,:,k);
    Fnew(:,:,k)=lyap(rho*inv(cuu(:,:,k)),cov_belief(:,:,k),-rho*U(:,:,k));
end

F_diff=Fnew-F;
end
function G=block_schr(V,W,gam,rho,blksize,sub_mat_size)
    G=V;
end
function [c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
        = cst_plattform_primal_diff(D,idx,b,u,c_bi,c_ui,...
        c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,lam,rho)
%     tic
lam_di=lam.lam_d;
% lam_b=lam.lam_b;
lam_up=lam.lam_up;
lam_c=lam.lam_c;
rho_d=rho.rho_d;
rho_up=rho.rho_up;

horizon = size(c_bi,2);
belief_dim = size(c_bi,1);
comp_amount = 2;
single_comp_dim = belief_dim / comp_amount;
ctrl_dim = 2;
stDim=2;
n_agent = size(D.Nodes,1);
for k=1:horizon-1
%     c_bi(:,k) = c_bi(:,k) + rho(1) * (x{idx}(:,k) - );
    uj_sum = zeros(ctrl_dim,1);
    for j = 2:n_agent-1
        uj_sum = uj_sum + u{j}(:,k);
    end   
    c_ui(5:6,k) = c_ui(5:6,k) + ...
        rho_up * 3*(3*u{idx}(5:6,k) - uj_sum + transpose(lam_up(1,:,k)));
    c_ui_ui(5:6,5:6,k) = c_ui_ui(5:6,5:6,k) + ...
        rho_up * 3 * 3 * eye(ctrl_dim);
end
[eid,nid] = inedges(D,idx);
for i_comp = 1:2
    for k=1:horizon
        
        for j_nid = 1:length(nid)-1
            j = nid(j_nid);
            edge_row = eid(j_nid);
            
            xj = b{j}(1:stDim,k);
            x_in_b = (i_comp-1)*single_comp_dim+3:(i_comp-1)*single_comp_dim+4;
            w1=b{idx}(21,k);
            w2=b{idx}(42,k);
            x_Plattform = b{idx}(x_in_b,k);
            formation_residue = (xj-x_Plattform-(D.Edges.nom_formation_2(edge_row,:))')*w2^2 ...
                +(xj-x_Plattform-(D.Edges.nom_formation_1(edge_row,:))')*w1^2;
            inc_c_bi = - (w1^2+w2^2)*(formation_residue + transpose(lam_di(j-1,:,k)));
            c_bi(x_in_b,k) = c_bi(x_in_b,k) + rho_d * inc_c_bi;
            c_bi_bi(x_in_b,x_in_b,k) = c_bi_bi(x_in_b,x_in_b,k) ...
                + rho_d * (w1^2+w2^2)^2 * eye(stDim);
        end

    end
end
for k=1:horizon
    components_amount=2;
    [x_idx, P_idx, w] = b2xPw(b{idx}(:,k), 4, components_amount);
    x_platf_weighted = zeros(2,components_amount);
    for i=1:components_amount
        x_platf_weighted(:,i)=transpose(x_idx{i}(3:4)*w(i));
    end
    x_platf= [sum(x_platf_weighted(1,:));sum(x_platf_weighted(2,:))];
    for j_nid = 1:length(nid)-1
        j = nid(j_nid);
        edge_row = eid(j_nid);

%         consensus_residue = b{j}(7:8,k)-x_platf;
%         inc_c_bi = - (consensus_residue + transpose(lam_b(j-1,:,k)));
%         x_in_b=3:4;
%         c_bi(x_in_b,k) = c_bi(x_in_b,k) + w(1)*rho_d * inc_c_bi;
%         c_bi_bi(x_in_b,x_in_b,k) = c_bi_bi(x_in_b,x_in_b,k) ...
%             + w(1)*rho_d * eye(stDim);
%         x_in_b=24:25;
%         c_bi(x_in_b,k) = c_bi(x_in_b,k) + w(2)* rho_d * inc_c_bi;
%         c_bi_bi(x_in_b,x_in_b,k) = c_bi_bi(x_in_b,x_in_b,k) ...
%             + w(2)* rho_d * eye(stDim);
    end

end
for k=horizon:horizon
    components_amount=2;
    x_goals = zeros(2,components_amount);

    [x_platf_comp, P_platf, w] = b2xPw(b{idx}(:,k), 4, components_amount);
    for i_comp = 1:components_amount
%         x_goal_in_b = (i_comp2-1)*single_comp_dim+1:(i_comp2-1)*single_comp_dim+2;
        x_goals(:,i_comp)=transpose(x_platf_comp{i_comp}(1:2));
    end
    x_opposite = b{5}(1:2,k);
    compl_residue = w(1)^2*(x_opposite-x_goals(:,2))+w(2)^2*(x_opposite-x_goals(:,1));
    for i_comp = 1:components_amount
        x_goal_in_b = (i_comp-1)*single_comp_dim+1:(i_comp-1)*single_comp_dim+2;
        w_in_b = i_comp*single_comp_dim;
        if i_comp ==1
            i_w = 2;
        else
            i_w = 1;
        end
        inc_c_bi = (compl_residue + transpose(lam_c(1,:,k)))*(-w(i_w)^2);
        c_bi(x_goal_in_b,k) = c_bi(x_goal_in_b,k) + 100*rho_d * inc_c_bi;
        c_bi_bi(x_goal_in_b,x_goal_in_b,k) = c_bi_bi(x_goal_in_b,x_goal_in_b,k) ...
            + 100*rho_d * eye(stDim) * w(i_w)^4;
        c_bi(w_in_b,k) = c_bi(w_in_b,k) + 100*rho_d * (compl_residue + transpose(lam_c(1,:,k)))'*(x_opposite-x_goals(:,i_w));
    end
end
% time_admm=toc
end
function [c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
        = cst_bypass_primal_diff(D,idx,b,u,last_u_from_com,c_bi,c_ui,...
        c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,lam,rho)
horizon = size(c_bi,2);
belief_dim = size(c_bi,1);
ctrl_dim = size(c_ui,1);
stDim=2;
n_agent = size(D.Nodes,1);
incoming_nbrs_idces = predecessors(D,idx)';
dim_xy = 2;
if rho.rho_w~=0
    for k=1:horizon-1
        sum_of_diff = zeros(dim_xy,1);
        for j=incoming_nbrs_idces
            last_com_t_k = 0.5*(last_u_from_com{idx}(:,k)+last_u_from_com{j}(:,k));
            sum_of_diff = sum_of_diff + u{idx}(:,k)-last_com_t_k;
        end
        c_ui(:,k) = c_ui(:,k) + lam.lam_w(idx,:,k)'+2*rho.rho_w * sum_of_diff;
%         if abs(rho_up*lam_w(idx,:,k)'-2*rho_d * sum_of_diff)>0.0001
%             a=1;
%         end
        c_ui_ui(:,:,k) = c_ui_ui(:,:,k) + 2*rho.rho_w * (length(incoming_nbrs_idces))*eye(ctrl_dim);
    end
end
% for k=1:horizon-1
% %     c_bi(:,k) = c_bi(:,k) + rho(1) * (x{idx}(:,k) - );
%     uj_sum = zeros(ctrl_dim,1);
%     for j = 2:n_agent
%         uj_sum = uj_sum + u{j}(:,k);
%     end   
%     c_ui(:,k) = c_ui(:,k) -rho_up * (3*u{1}(5:6,k) - uj_sum + transpose(lam_up(1,:,k)));
%     c_ui_ui(:,:,k) = c_ui_ui(:,:,k) + rho_up * eye(ctrl_dim);
% end
% for k=horizon:horizon
%     components_amount=2;
%     stDim_platf = 4;
%     [x_platf_comp, P_platf, w] = b2xPw(b{1}(:,k), stDim_platf, components_amount);
% 
%     x_platf_weighted = zeros(2,components_amount);
%     x_goals = zeros(2,components_amount);
%     for i=1:components_amount
%         x_platf_weighted(:,i)=transpose(x_platf_comp{i}(3:4)*w(i));
%         x_goals(:,i)=transpose(x_platf_comp{i}(1:2));
%     end
%     x_platf= [sum(x_platf_weighted(1,:));sum(x_platf_weighted(2,:))];
%     edge_row = idx-1;
%     x_idx = b{idx}(1:stDim,k);
% %     formation_residue = b{idx}(1:stDim,k)-x_platf-(D.Edges.nom_formation_2(edge_row,:))';
%     compl_residue = w(1)^2*(x_idx-x_goals(:,2))+w(2)^2*(x_idx-x_goals(:,1));
% 
%     x_in_b = 1:2;
%     inc_c_bi = (compl_residue + transpose(lam_w(1,:,k)))*(w(1)^2+w(2)^2);
%     c_bi(x_in_b,k) = c_bi(x_in_b,k) ...
%                 + 100*rho_d * inc_c_bi;
%     c_bi_bi(x_in_b,x_in_b,k) = c_bi_bi(x_in_b,x_in_b,k) ...
%                 + 100*rho_d * (w(1)^2+w(2)^2)^2*eye(stDim);
% end
end

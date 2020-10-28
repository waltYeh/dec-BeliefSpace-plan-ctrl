function [c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
        = cst_assist_primal_diff(D,idx,b,u,c_bi,c_ui,...
        c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,lam,rho)
lam_di=lam.lam_d;
% lam_b=lam.lam_b;
lam_up=squeeze(lam.lam_up);
% lam_w=lam.lam_w;
rho_d=rho.rho_d;
rho_up=rho.rho_up;
horizon = size(c_bi,2);
belief_dim = size(c_bi,1);
ctrl_dim = size(c_ui,1);
stDim=2;
n_agent = size(D.Nodes,1);
c_ui_inc = zeros(size(c_ui));
u_residue = zeros(ctrl_dim,horizon);
for k=1:horizon-1
%     c_bi(:,k) = c_bi(:,k) + rho(1) * (x{idx}(:,k) - );
    uj_sum = zeros(ctrl_dim,1);
    for j = 2:n_agent-1
        uj_sum = uj_sum + u{j}(:,k);
    end   
    u_residue(:,k) = 3*u{1}(5:6,k) - uj_sum;
    c_ui_inc(:,k) = -rho_up * (u_residue(:,k) + lam_up(:,k));
end
for k=1:horizon-1
    c_ui(:,k) = c_ui(:,k) + c_ui_inc(:,k);
    c_ui_ui(:,:,k) = c_ui_ui(:,:,k) + rho_up * eye(ctrl_dim);
%     figure(122)
%     quiver(b{idx}(1,k),b{idx}(2,k),c_ui_inc(1,k),c_ui_inc(2,k))
%     hold on
%     axis equal
end


for k=1:horizon
    components_amount=2;
    stDim_platf = 4;
    [x_platf_comp, P_platf, w] = b2xPw(b{1}(:,k), stDim_platf, components_amount);

    x_platf_weighted = zeros(2,components_amount);
    for i=1:components_amount
        x_platf_weighted(:,i)=transpose(x_platf_comp{i}(3:4)*w(i));
    end
    x_platf= [sum(x_platf_weighted(1,:));sum(x_platf_weighted(2,:))];
    edge_row = idx-1;
    formation_residue = (b{idx}(1:stDim,k)-x_platf-(D.Edges.nom_formation_2(edge_row,:))')*w(2)^2 ...
        +(b{idx}(1:stDim,k)-x_platf-(D.Edges.nom_formation_1(edge_row,:))')*w(1)^2;
    x_in_b = 1:2;
    inc_c_bi = (w(1)^2+w(2)^2)*(formation_residue + transpose(lam_di(idx-1,:,k)));
    c_bi(x_in_b,k) = c_bi(x_in_b,k) ...
                + rho_d * inc_c_bi;
    c_bi_bi(x_in_b,x_in_b,k) = c_bi_bi(x_in_b,x_in_b,k) ...
                + rho_d * (w(1)^2+w(2)^2)^2 * eye(stDim);
            
%     consensus_residue = b{idx}(7:8,k)-x_platf;
%     inc_c_bi = (consensus_residue + transpose(lam_b(idx-1,:,k)));
%     x_in_b = 7:8;
%     c_bi(x_in_b,k) = c_bi(x_in_b,k) + rho_d * inc_c_bi;
%     c_bi_bi(x_in_b,x_in_b,k) = c_bi_bi(x_in_b,x_in_b,k) ...
%             + rho_d * eye(stDim);
end
end
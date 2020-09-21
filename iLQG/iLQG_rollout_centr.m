function [x,cost_new] = iLQG_rollout_centr(DYNCST, x0, u)

        [x,un,cost_new]  = forward_pass(x0,u,DYNCST);

function [xnew,unew,cnew] = forward_pass(x0,u,DYNCST)
% l (Schwarting j_k^i) is taken into the function as the argument du
% parallel forward-pass (rollout)
% internally time is on the 3rd dimension, 
% to facillitate vectorized dynamics calls
% Alpha is a series of possible step length, 
% it is multiplied with du(:,i) in parallel
n        = size(x0,1);
K        = 1;
K1       = ones(1,K); % useful for expansion
m        = size(u,1);
N        = size(u,2);

xnew        = zeros(n,K,N);
xnew(:,:,1) = x0(:,ones(1,K));
unew        = zeros(m,K,N);
cnew        = zeros(1,K,N+1);
for i = 1:N
    unew(:,:,i) = u(:,i*K1);

    [xnew(:,:,i+1), cnew(:,:,i)]  = DYNCST(xnew(:,:,i), unew(:,:,i), i*K1);
end
[~, cnew(:,:,N+1)] = DYNCST(xnew(:,:,N+1),nan(m,K,1),i);

% components_amount = 2;
% horiz = N+1;
% mu = cell(components_amount,1);
% sig = cell(components_amount,1);
% weight = zeros(components_amount,horiz);
% for i_comp = 1:components_amount
%     mu{i_comp}=zeros(4,horiz);
%     sig{i_comp}=zeros(16,horiz);
% %     weight(i_comp,i)=w(i_comp);
% end
% for i=1:horiz
%     [x, P, w] = b2xPw(xnew(:,:,i), 4, components_amount);
%     for i_comp = 1:components_amount
%         mu{i_comp}(:,i)=x{i_comp};
%         sig{i_comp}(:,i)=P{i_comp}(:);
%         weight(i_comp,i)=w(i_comp);
%     end
% end
% figure(11)
% hold on
% for i_comp = 1:components_amount
%     plot(mu{i_comp}(1,:),mu{i_comp}(2,:),'.')
%     axis equal
%     plot(mu{i_comp}(3,:),mu{i_comp}(4,:),'+')
% end
% 
% for k=1:horiz
%     pointsToPlot = drawResultGMM([mu{1}(:,k); sig{1}(:,k)], 4);
%     figure(8)
%     plot(pointsToPlot(1,:),pointsToPlot(2,:),'b')
%     pointsToPlot = drawResultGMM([mu{2}(:,k); sig{2}(:,k)], 4);
%     plot(pointsToPlot(1,:),pointsToPlot(2,:),'r')
%     pause(0.1);
% end
% hold off

% put the time dimension in the columns
xnew = permute(xnew, [1 3 2]);
unew = permute(unew, [1 3 2]);
cnew = permute(cnew, [1 3 2]);

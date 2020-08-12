function nSigma = sigmaToCollide_multiagent_D(D,idx,b,stDim,stateValidityChecker)
%%%%%%%%%%%%%%%%%%%%%%%
% Compute std devs to collision based on 
% Section 5 of Van den Berg et al. IJRR 2012
%
% Inputs:
%   b: belief state
%   stateValidityChecker: function for checking collision
%
% Output:
%   nSigma: number of std devs robot should deviate to collide
%%%%%%%%%%%%%%%%%%%%%%
rj_orig = 0.2;
ri_orig = 0.2;
incoming_nbrs_idces = predecessors(D,idx)';
% global ROBOT_RADIUS
% R_orig =  ROBOT_RADIUS; % save robot radius
% if size(b,1)==1
    % this means we are using finite difference
    %bj 1~3x6x427
    nSigma = 3.5*ones(size(D.Nodes,1),size(b,3));
    for j = incoming_nbrs_idces
        for k = 1:size(b,3)
            xi = b(idx,1:stDim,k)';
            xj = b(j,1:stDim,k)';
            Pi = zeros(stDim, stDim); % covariance matrix
            Pj = zeros(stDim, stDim);
            % Extract columns of principal sqrt of covariance matrix
            % right now we are not exploiting symmetry
            for d = 1:stDim
                Pi(:,d) = b(idx,d*stDim+1:(d+1)*stDim, k);
                Pj(:,d) = b(j,d*stDim+1:(d+1)*stDim, k);
            end

            eigval_i = eig(Pi); % get eigen values
            eigval_j = eig(Pj);
            lambda_i = max(eigval_i); % get largest eigen val
            lambda_j = max(eigval_j);
            di = sqrt(lambda_i); % distance along 1 std dev  
            dj = sqrt(lambda_j);
            % number of standard deviations at which robot collides
            % at s = 0, f goes to infinite so not good -> better to use small value of 0.01
            for s = 0.01:0.2:2.21

                % inflate robot radius 
                ri = ri_orig + s*di;
                rj = rj_orig + s*dj;

                % if robot collided
                if stateValidityChecker(xi,ri, xj, rj) == 0    

                    nSigma(j,k) = s;     
                    nSigma(idx,k) = s;
                    break;
                end
            end
        end

    end
%     nSigma = nSigma;
% else
%     nSigma = 3.5*ones(size(D.Nodes,1),size(bi,2));
% 
%     for k = 1:size(bi,2)
% 
%         xi = bi(1:stDim,k);
%         xj = bj(1:stDim,k);
%         Pi = zeros(stDim, stDim); % covariance matrix
%         Pj = zeros(stDim, stDim);
%         % Extract columns of principal sqrt of covariance matrix
%         % right now we are not exploiting symmetry
%         for d = 1:stDim
%             Pi(:,d) = bi(d*stDim+1:(d+1)*stDim, k);
%             Pj(:,d) = bj(d*stDim+1:(d+1)*stDim, k);
%         end
% 
%         eigval_i = eig(Pi); % get eigen values
%         eigval_j = eig(Pj);
%         lambda_i = max(eigval_i); % get largest eigen val
%         lambda_j = max(eigval_j);
%         di = sqrt(lambda_i); % distance along 1 std dev  
%         dj = sqrt(lambda_j);
%         % number of standard deviations at which robot collides
%         % at s = 0, f goes to infinite so not good -> better to use small value of 0.01
%         for s = 0.01:0.2:3.21
% 
%             % inflate robot radius 
%             ri = ri_orig + s*di;
%             rj = rj_orig + s*dj;
% 
%             % if robot collided
%             if stateValidityChecker(xi,ri, xj, rj) == 0    
% 
%                 nSigma(k) = s;            
% 
%                 break;
%             end
%         end
% 
% 
%     end
% 
% % ROBOT_RADIUS = R_orig; % reset robot radius
% end
end
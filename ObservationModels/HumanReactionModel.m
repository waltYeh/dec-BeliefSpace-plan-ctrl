%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Belief Space Planning with iLQG
% Copyright 2017
% Author: Saurav Agarwal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Measure signal to beacons, signal strength falls quadratically with distance.
% Robot can see all beacons
classdef HumanReactionModel < ObservationModelBase
    
    properties(Constant = true)
        obsDim = 4;
        obsNoiseDim = 4;
    end
    
    properties
%         sigma_b = diag([0.02,0.02,20.0,20.0]); 
        R_true = diag([0.1, 0.1, 0.001, 0.001]);
        R_speed = diag([0.01, 0.01]);

        R_est = diag([0.1, 0.1, 0.01, 0.01]);
        R_w = diag([4,2,2, 2]);
    end
    
    methods
        
        function obj = HumanReactionModel()
            obj@ObservationModelBase();                                  
        end
        
        function z = getObservation(obj, x, varargin)
            % getObservation, Get what sensor sees conditioned on varargin.            
            % getObservation(obj, x) all visible features with noise
            % getObservation(obj, x, 'nonoise') all visible features with no noise measurements
            % Output :
            % z Observation vector
            % idfs Ids of observations
            
            x_target = x(1); 
            y_target = x(2); 
            x_man = x(3); 
            y_man = x(4); 
            v_max=3;
            k_factor = 2;
            pos_error=norm([x_target;y_target]-[x_man;y_man]);
            speed_chase=v_max*2/pi*atan(k_factor*pos_error);
            direction=atan2(y_target-y_man,x_target-x_man);
            
%             range = obj.computeRange(x);                        
                                                          
            if nargin == 2 % noisy observations
                v = obj.computeObservationNoise();
                z = [speed_chase;
                    direction;
                    x_man;
                    y_man] + v;
                
            elseif nargin > 2 && strcmp('nonoise',varargin{1}) == 1 
                % nonoise
                z=[speed_chase;
                    direction;
                    x_man;
                    y_man];
            elseif nargin > 2 && strcmp('truenoise',varargin{1}) == 1 
                v = obj.computeObservationNoiseTrue(x);
                z = [speed_chase;
                    direction;
                    x_man;
                    y_man] + v;
            else                
                error('unknown inputs')                
            end
            
        end
        
        function H = getObservationJacobian(obj,x, v)
            % Compute observation model jacobian w.r.t state
            % Inputs:
            % x: robot state
            % f: feature pose
            % Outputs:
            % H: Jacobian matrix
            
            x_target = x(1); 
            y_target = x(2); 
            x_man = x(3); 
            y_man = x(4); 
            v_max=3;
            k_factor = 2;
            pos_error = norm([x_target;y_target]-[x_man;y_man]);
            speed_chase=v_max*2/pi*atan(k_factor*pos_error);
            direction=atan2(y_target-y_man,x_target-x_man);
            tmp_2_3_K_dist_3_2_2 = v_max*4/pi/(1+4*pos_error^2)*2/3*pos_error^(3/2);
            dv_dx1 = tmp_2_3_K_dist_3_2_2*2*(x(1)-x(3));
            dv_dx2 = tmp_2_3_K_dist_3_2_2*2*(x(2)-x(4));
            dv_dx3 = tmp_2_3_K_dist_3_2_2*2*(x(3)-x(1));
            dv_dx4 = tmp_2_3_K_dist_3_2_2*2*(x(4)-x(2));
            % sin_th = sin(direction);
            cos_th = cos(direction);
            tmp_datanw_dw=cos_th^2;
            tmp_datanw_dw_x1_x3_2=tmp_datanw_dw/(x(1)-x(3))^2;
            % tmp_sin_cos_2_x1_x3_2 = -sin(direction)*cos(direction)^2/(x(1)-x(3))^2;
            % tmp_cos_cos_2_x1_x3_2 = cos(direction)*cos(direction)^2/(x(1)-x(3))^2;
            dth_dx1 = tmp_datanw_dw_x1_x3_2 * (x(4)-x(2));
            dth_dx2 = tmp_datanw_dw_x1_x3_2 * (x(1)-x(3));
            dth_dx3 = tmp_datanw_dw_x1_x3_2 * (x(2)-x(4));
            dth_dx4 = tmp_datanw_dw_x1_x3_2 * (x(3)-x(1));
            % d_sin_dx1 = tmp_cos_cos_2_x1_x3_2 * (x(4)-x(2));
            % d_sin_dx2 = tmp_cos_cos_2_x1_x3_2 * (x(1)-x(3));
            % d_sin_dx3 = tmp_cos_cos_2_x1_x3_2 * (x(2)-x(4));
            % d_sin_dx4 = tmp_cos_cos_2_x1_x3_2 * (x(3)-x(1));

            % dy3_dx1=K*2/3*distance^(3/2)*2*(x(1)-x(3))*cos(direction)...
            %     -sin(direction)*cos(direction)^2*(x(4)-x(2))/(x(1)-x(3))^2*speed_chase;
            % dy3_dx2=K*2/3*distance^(3/2)*2*(x(2)-x(4))*cos(direction)...
            %     -sin(direction)*cos(direction)^2*(x(1)-x(3))/(x(1)-x(3))^2*speed_chase;
            % dy3_dx3=K*2/3*distance^(3/2)*2*(x(3)-x(1))*cos(direction)...
            %     -sin(direction)*cos(direction)^2*(x(2)-x(4))/(x(1)-x(3))^2*speed_chase;
            % dy3_dx4=K*2/3*distance^(3/2)*2*(x(4)-x(2))*cos(direction)...
            %     -sin(direction)*cos(direction)^2*(x(3)-x(1))/(x(1)-x(3))^2*speed_chase;
            % dy1_dx1=dv_dx1*cos_th+d_cos_dx1*speed_chase;
            % dy1_dx2=dv_dx2*cos_th+d_cos_dx2*speed_chase;
            % dy1_dx3=dv_dx3*cos_th+d_cos_dx3*speed_chase;
            % dy1_dx4=dv_dx4*cos_th+d_cos_dx4*speed_chase;
            % dy2_dx1=dv_dx1*sin_th+d_sin_dx1*speed_chase;
            % dy2_dx2=dv_dx2*sin_th+d_sin_dx2*speed_chase;
            % dy2_dx3=dv_dx3*sin_th+d_sin_dx3*speed_chase;
            % dy2_dx4=dv_dx4*sin_th+d_sin_dx4*speed_chase;
            % dy_dx=[0,0,0,0;
            %     %dy1_dx1,dy1_dx2,dy1_dx3,dy1_dx4;
            %      0,0,0,0;
            %      %dy2_dx1,dy2_dx2,dy2_dx3,dy2_dx4;
            %     0,0,1,0;
            %     0,0,0,1];
            % dy2_dx4=dv_dx4*sin_th+d_sin_dx4*speed_chase;
            H=[dv_dx1,dv_dx2,dv_dx3,dv_dx4;
                 dth_dx1,dth_dx2,dth_dx3,dth_dx4;
                0,0,1,0;
                0,0,0,1];
        end
        
        function M = getObservationNoiseJacobian(obj,x)%,v,z)
            n = 4;%length(z);
            M = 8/(1/(norm(x(4)-[1.5])^2+0.05) + 1/norm(x(3)-[1])^2 + 1) * eye(n);
%             M(1,1) = 1;
%             M(2,2) = 1;
        end
        
        function R = getObservationNoiseCovariance(obj,x,z)
%             noise_std = repmat(obj.sigma_b,size(z,1),1);
%             
%             R = diag(noise_std.^2);
            R = obj.R_true;
            
        end                                                      
        
        function v = computeObservationNoiseTrue(obj,x)
            
            noise_std = chol(obj.R_true)';
            
            v = obj.getObservationNoiseJacobian(x)*noise_std*randn(obj.obsNoiseDim,1);
        end
        
        function v = computeObservationNoise(obj)
            
            noise_std = repmat(obj.R_est,size(obj.obsNoiseDim,1),1);
            
            v = randn(size(obj.obsNoiseDim,1),1).*noise_std;
        end
        
        function innov = computeInnovation(obj,Xprd,Zg)
                        
            z_prd = obj.getObservation(Xprd, 'nonoise');
            
            innov = Zg - z_prd;
        end
        
%         function range = computeRange(obj, x)
%             
%             % Compute exact observation
%             dx= obj.landmarkPoses(1,:) - x(1);
%             dy= obj.landmarkPoses(2,:) - x(2);
%             
%             range = sqrt(dx.^2 + dy.^2);                
%             
%             range = range';
%          end
        
    end
    
end

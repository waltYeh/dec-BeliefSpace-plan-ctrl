%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Belief Space Planning with iLQG
% Copyright 2017
% Author: Saurav Agarwal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2D Point robot with additive process noise.
classdef HumanMind < MotionModelBase
    properties (Constant = true) % note that all properties must be constant, because we have a lot of copies of this object and it can take a lot of memory otherwise.
        stDim = 4; % state dimension
        ctDim = 4;  % control vector dimension
        wDim = 4;   % Process noise (W) dimension
        P_Wg = diag([0.001,0.001,20.0,20.0]);
%         P_Wg = diag([0.005,0.005,0.005,0.005].^2); % covariance of state-additive-noise
        sigma_b_u = [0.0;0.0;0.0;0.0]; % A constant bias intensity (std dev) of the control noise
        eta_u = [0;0;0;0]; % A coefficient, which makes the control noise intensity proportional to the control signal       
        zeroNoise = [0;0;0;0]; 
        ctrlLim = [-25.0, 25.0;-25.0, 25.0; -5.0,5.0;-5.0,5.0]; % max control for Vx and Vy
    end
    
    methods
        
        function obj = HumanMind(dt)
            obj@MotionModelBase();      
            obj.dt = dt;
        end
        
        function x_next = evolve(obj,x,u,w) % discrete motion model equation  
            for i=1:obj.ctDim
                if u(i) < obj.ctrlLim(i,1)
                    u(i) = obj.ctrlLim(i,1);                
                elseif u(i) > obj.ctrlLim(i,2)
                    u(i) = obj.ctrlLim(i,2);
                end
            end
            ux_target = u(1); 
            uy_target = u(2); 
            ux_wheelchair = u(3); 
            uy_wheelchair = u(4); 
            x_target = x(1); 
            y_target = x(2);
            x_man = x(3); 
            y_man = x(4); 
            
            K=3;
            pos_error=norm([x_target;y_target]-[x_man;y_man]);
            speed_chase=K*2/pi*atan(2*pos_error);
            direction=atan2(y_target-y_man,x_target-x_man);
            
            vx_manpower = speed_chase * cos(direction);
            vy_manpower = speed_chase * sin(direction);
            vx_manpower = 0;
            vy_manpower = 0;
            x_next=[x_target+ux_target*obj.dt;
                y_target+uy_target*obj.dt;
                x_man+(vx_manpower + ux_wheelchair)*obj.dt;
                y_man+(vy_manpower + uy_wheelchair)*obj.dt];

            x_next = x_next + sqrt(obj.dt)*w;
        end
        
        function A = getStateTransitionJacobian(obj,x,u,w) % state Jacobian
            x_target = x(1); 
            y_target = x(2);
            x_man = x(3); 
            y_man = x(4); 
            K=3;
            pos_error=norm([x_target;y_target]-[x_man;y_man]);
            speed_chase=K*2/pi*atan(2*pos_error);
            direction=atan2(y_target-y_man,x_target-x_man);
            
            tmp_2_3_K_dist_3_2_2 = K*4/pi/(1+4*pos_error^2)*2/3*pos_error^(3/2);
            dv_dx1 = tmp_2_3_K_dist_3_2_2*2*(x(1)-x(3));
            dv_dx2 = tmp_2_3_K_dist_3_2_2*2*(x(2)-x(4));
            dv_dx3 = tmp_2_3_K_dist_3_2_2*2*(x(3)-x(1));
            dv_dx4 = tmp_2_3_K_dist_3_2_2*2*(x(4)-x(2));
            
            sin_th = sin(direction);
            cos_th = cos(direction);
            tmp_datanw_dw=cos_th^2;
            tmp_datanw_dw_x1_x3_2=tmp_datanw_dw/(x(1)-x(3))^2;
       
            dth_dx1 = tmp_datanw_dw_x1_x3_2 * (x(4)-x(2));
            dth_dx2 = tmp_datanw_dw_x1_x3_2 * (x(1)-x(3));
            dth_dx3 = tmp_datanw_dw_x1_x3_2 * (x(2)-x(4));
            dth_dx4 = tmp_datanw_dw_x1_x3_2 * (x(3)-x(1));
            
%             tmp_sin_cos_2_x1_x3_2 = -sin(direction)*cos(direction)^2/(x(1)-x(3))^2;
%             tmp_cos_cos_2_x1_x3_2 = cos(direction)*cos(direction)^2/(x(1)-x(3))^2;
            
            d_sin_dx1 = dth_dx1 * cos_th;
            d_sin_dx2 = dth_dx2 * cos_th;
            d_sin_dx3 = dth_dx3 * cos_th;
            d_sin_dx4 = dth_dx4 * cos_th;
            
            d_cos_dx1 = dth_dx1 * (-sin_th);
            d_cos_dx2 = dth_dx2 * (-sin_th);
            d_cos_dx3 = dth_dx3 * (-sin_th);
            d_cos_dx4 = dth_dx4 * (-sin_th);

            df3_dx1=dv_dx1*cos_th+d_cos_dx1*speed_chase;
            df3_dx2=dv_dx2*cos_th+d_cos_dx2*speed_chase;
            df3_dx3=dv_dx3*cos_th+d_cos_dx3*speed_chase;
            df3_dx4=dv_dx4*cos_th+d_cos_dx4*speed_chase;
            df4_dx1=dv_dx1*sin_th+d_sin_dx1*speed_chase;
            df4_dx2=dv_dx2*sin_th+d_sin_dx2*speed_chase;
            df4_dx3=dv_dx3*sin_th+d_sin_dx3*speed_chase;
            df4_dx4=dv_dx4*sin_th+d_sin_dx4*speed_chase;
%             A = [1,0,0,0;
%                 0,1,0,0;
%                 df3_dx1, df3_dx2, df3_dx3 + 1, df3_dx4;
%                 df4_dx1, df4_dx2, df4_dx3, df4_dx4 + 1];
            A = eye(4);
        end
        
        function B = getControlJacobian(obj,x,u,w) % control Jacobian
            B = obj.dt*eye(4);
        end
        
        function G = getProcessNoiseJacobian(obj,x,u,w) % noise Jacobian
            G = sqrt(obj.dt)*eye(4);
        end
        
        function Q = getProcessNoiseCovariance(obj,x,u) % compute the covariance of process noise based on the current poistion and controls
            Q = obj.P_Wg;
        end
        
        function w = generateProcessNoise(obj,x,u) % simulate (generate) process noise based on the current state and controls
            w = mvnrnd(zeros(obj.stDim,1),obj.P_Wg)';
            %multi variant random (mu,sigma)
        end
        
        function U = generateOpenLoopControls(obj,x0,xf)                                                          
            
            d = xf - x0; % displacement
            
            % cannot drive at control limit because then there is no room
            % for optimizer to add deltas to control input
            maxVx = 0.5*max(obj.ctrlLim(1,:)); % max speed at which robot can go
            maxVy = 0.5*max(obj.ctrlLim(2,:)); % max speed at which robot can go
            
            nDT = ceil(max(abs(d(1))/ (maxVx*obj.dt), abs(d(2))/ (maxVy*obj.dt)));
            % time steps required to go from x0 to xf
            
            U = repmat((xf-x0)/(nDT*obj.dt),1,nDT);
            % create a 1xnDT tiling of control vector
            
        end
        
    end
end
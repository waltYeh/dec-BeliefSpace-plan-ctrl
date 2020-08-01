%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Belief Space Planning with iLQG
% Copyright 2017
% Author: Saurav Agarwal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Measure range to beacons
% Robot can see all beacons
classdef TwoDSimpleObsModel < ObservationModelBase
    
    properties(Constant = true)
        obsDim = 2;
        obsNoiseDim = 2;
    end
    
    properties
%         sigma_b = 0.1; 
%         eta = 0.01;
        R_true = diag([0.001, 0.001]);
%         R_speed = diag([0.01, 0.01]);
        R_est = diag([0.001, 0.001]);
    end
    
    methods
        function obj = TwoDSimpleObsModel()
            obj@ObservationModelBase();                                  
        end
        
        function z = getObservation(obj, x, varargin)
            % getObservation, Get what sensor sees conditioned on varargin.            
            % getObservation(obj, x) all visible features with noise
            % getObservation(obj, x, 'nonoise') all visible features with no noise measurements
            % Output :
            % z Observation vector
            % idfs Ids of observations
            
            % if there are no landmarks to see
%             if isempty(obj.landmarkPoses) == 1               
%                 error('There are no landmarks to see');                                
%             end
%             
%             range = obj.computeRange(x);                        
%                                                   
%             x_1 = x(1); 
%             x_2 = x(2); 
            if nargin == 2 % noisy observations
                v = obj.computeObservationNoise(range);
                z = [x(1);
                    x(2)]; + v;
                
            elseif nargin > 2 && strcmp('nonoise',varargin{1}) == 1 % nonoise
                z = [x(1);
                    x(2)];
            elseif nargin > 2 && strcmp('truenoise',varargin{1}) == 1 
                v = obj.computeObservationNoiseTrue();
                z = [x(1);
                    x(2)] + v;
            else                
                error('unknown inputs')                
            end
            
        end
        function v = computeObservationNoiseTrue(obj)
            
            noise_std = chol(obj.R_true)';
            
            v = noise_std*randn(obj.obsNoiseDim,1);
        end
        function H = getObservationJacobian(obj,x, v)
            % Compute observation model jacobian w.r.t state
            % Inputs:
            % x: robot state
            % f: feature pose
            % Outputs:
            % H: Jacobian matrix
            H = eye(2);
%             dx = x(1) - obj.landmarkPoses(1,:);
%             dy = x(2) - obj.landmarkPoses(2,:);
%             
%             r = sqrt(dx.^2 + dy.^2);                         
%             
%             H = zeros(size(r,2),2);
%             
%             for i = 1:size(r,2)
%                 H(i,:) = [dx(i)/r(i) dy(i)/r(i)];
%             end            
%             
        end
        
        function M = getObservationNoiseJacobian(obj,x,v,z)
            M = eye(2);%diag(z);
        end
        
        function R = getObservationNoiseCovariance(obj,x,z)
            R = obj.R_true;
            
        end                                                      
        
        function v = computeObservationNoise(obj,z)
            
            noise_std = repmat(obj.R_est,size(z,1),1);
            
            v = randn(size(obj.obsNoiseDim,1),1).*noise_std;
        end
        
        function innov = computeInnovation(obj,Xprd,Zg)
                        
            z_prd = obj.getObservation(Xprd, 'nonoise');
            
            innov = Zg - z_prd;
        end
        
        function range = computeRange(obj, x)
            
            % Compute exact observation
            dx= obj.landmarkPoses(1,:) - x(1);
            dy= obj.landmarkPoses(2,:) - x(2);
            
            range = sqrt(dx.^2 + dy.^2);                
            
            range = range';
         end
        
    end
    
end

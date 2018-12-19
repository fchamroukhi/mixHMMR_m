function [tau_tk, xi_tkl, alpha_tk, beta_tk, loglik, xi_summed] = forwards_backwards(prior, transmat, f_tk)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [tau_tk, xi_tlk, alpha_tk, beta_tk, loglik, xi_summed] = forwards_backwards(prior, transmat, f_tk)
% forwards_backwards : calculates the E-step of the EM algorithm for an HMM
% (Gaussian HMM)

% Inputs :
%
%         prior(k) = Pr(z_1 = k)
%         transmat(\ell,k) = Pr(z_t=k | z_{t-1} = \ell)
%         f_tk(t,k) = Pr(y_t | z_y=k;\theta) %gaussian
%
% Outputs:
%
%        tau_tk(t,k) = Pr(z_t=k | X): post probs (smoothing probs)
%        xi_tk\elll(t,k,\ell)  = Pr(z_t=k, z_{t-1}=\ell | Y) t =2,..,n
%        with Y = (y_1,...,y_n);
%        alpha_tk(k,t): [Kxn], forwards probs: Pr(y1...yt, zt=k)
%        beta_tk(k,t): [Kxn], backwards probs: Pr(yt+1...yn|zt=k)
%
%
%
% Faicel Chamroukhi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 6, filter_only = 0; end

T = size(f_tk, 2);
K = length(prior);

if size(prior,2)~=1
    prior = prior';
end
scale = ones(1,T);%pour que loglik = sum(log(scale)) part de zero

prior = prior(:);
tau_tk = zeros(K,T);
xi_tkl = zeros(K,K,T-1);
xi_summed = zeros(K,K);
alpha_tk = zeros(K,T);
beta_tk = zeros(K,T);

%% forwards: calculate the alpha_tk's
t = 1;
alpha_tk(:,1) = prior.* f_tk(:,t);
[alpha_tk(:,t), scale(t)] = normalize(alpha_tk(:,t));

for t=2:T
    [alpha_tk(:,t), scale(t)] = normalize((transmat'*alpha_tk(:,t-1)) .* f_tk(:,t));
    %     filtered_prob (:,:,t-1)= normalize((transmat'*alpha(:,t-1)).*fit(:,t));%
    % NB: filtered_prob(t,k,l) => filter_prb = squeeze(sum(filters_prob, 2));
end
% %loglik (technique du scaling)
loglik = sum(log(scale));

if filter_only
    beta_tk = [];
    tau_tk = alpha_tk;
    return;
end
%% backwards: calculate the beta_tk's, the tau_tk's (and the xi_tkl's)
%t=T;

beta_tk(:,T) = ones(1,K);
tau_tk(:,T) = normalize(alpha_tk(:,T) .* beta_tk(:,T));

for t=T-1:-1:1
    beta_tk(:,t) = normalize(transmat * (beta_tk(:,t+1) .* f_tk(:,t+1)));
    tau_tk(:,t) = normalize(alpha_tk(:,t) .* beta_tk(:,t));
    xi_tkl(:,:,t) = normalize((transmat .* (alpha_tk(:,t) * (beta_tk(:,t+1) .* f_tk(:,t+1))')));
    xi_summed = xi_summed + sum(xi_tkl(:,:,t),3);
end


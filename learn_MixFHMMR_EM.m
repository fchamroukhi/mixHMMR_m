function MixFHMMR =  learn_MixFHMMR_EM(data, K, R, p, ...
    variance_type, ordered_states, total_EM_tries, max_iter_EM, init_kmeans, threshold, verbose)
%   MixFHMMR =  seq_clust_MixFHMMR(data, K, R, p,fs, variance_type,...
%               order_constraint, total_EM_tries, max_iter_EM, init_kmeans, threshold, verbose)
% Learn a mixture of Hidden Markov Moedel Regression for curve clustering by EM
%
%
% Inputs : 
%
%          1. data :  n curves each curve is composed of m points : dim(Y)=[n m] 
%                * Each curve is observed during the interval [0,T]=[t_1,...,t_m]
%                * t{j}-t_{j-1} = 1/fs (fs: sampling period)    
%          2. K: number of clusters
%          3. R: Number of polynomial regression components (regimes)
%          4. p: degree of the polynomials
% Options:
%          1. order_constraint: set to one if ordered segments (by default 0)
%          2. variance_type of the poynomial models for each cluster (free or
%          common, by defalut free)
%          3. init_kmeans: initialize the curve partition by Kmeans
%          4. total_EM_tries :  (the solution providing the highest log-lik is chosen
%          5. max_iter_EM
%          6. threshold: by defalut 1e-6
%          7. verbose : set to 1 for printing the "complete-log-lik"  values during
%          the EM iterations (by default verbose_EM = 0)
%
% Outputs : 
%
%          MixFHMMR : structure containing the following fields:
%                   
%          1. param : a structure containing the model parameters
%                       ({Wk},{alpha_k}, {beta_kr},{sigma_kr}) for k=1,...,K and k=1...R. 
%              1.1 Wk = (Wk1,...,w_kR-1) parameters of the logistic process:
%                  matrix of dimension [(q+1)x(R-1)] with q the order of logistic regression.
%              1.2 beta_k = (beta_k1,...,beta_kR) polynomial regression coefficient vectors: matrix of
%                  dimension [(p+1)xR] p being the polynomial  degree.
%              1.3 sigma_k = (sigma_k1,...,sigma_kR) : the variances for the R regmies. vector of dimension [Rx1]
%              1.4 pi_jkr :logistic proportions for cluster g
%
%          2. paramter_vector: parameter vector of the model: Psi=({Wg},{alpha_k},{beta_kr},{sigma_kr}) 
%                  column vector of dim [nu x 1] with nu = nbr of free parametres
%          3. h_ik = prob(curve|cluster_k) : post prob (fuzzy segmentation matrix of dim [nxK])
%          4. c_ik : Hard partition obtained by the AP rule :  c_{ik} = 1
%                    if and only c_i = arg max_k h_ik (k=1,...,K)
%          5. klas : column vector of cluster labels
%          6. tau_ijkr prob(y_{ij}|kth_segment,cluster_k), fuzzy
%          segmentation for the cluster g. matrix of dimension
%          [nmxR] for each g  (g=1,...,K).
%          7. Ex_k: curve expectation: sum of the polynomial components beta_kr ri weighted by 
%             the logitic probabilities pij_kr: Ex_k(j) = sum_{k=1}^R pi_jkr beta_kr rj, j=1,...,m. Ex_k 
%              is a column vector of dimension m for each g.
%          8. loglik : at convergence of the EM algo
%          9. stored_com-loglik : vector of stored valued of the
%          comp-log-lik at each EM teration 
%          
%          10. BIC value = loglik - nu*log(nm)/2.
%          11. ICL value = comp-loglik_star - nu*log(nm)/2.
%          12. AIC value = loglik - nu.
%          13. log_alphag_fg_xij 
%          14. polynomials 
%          15. weighted_polynomials 
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi (septembre 2009) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off

[n, m] = size(data);%n  curves of m observations
% regression matrix
t=0:m-1;
%t= linspace(0,1,m);
[phi] = designmatrix(t,p);% pour 1 courbe
X = repmat(phi,n,1);% pour les n courbes
%
Y=reshape(data',[],1); 


% % main algorithm
try_EM = 0; 
best_loglik = -inf;
cputime_total = [];
while (try_EM < total_EM_tries)
    try_EM = try_EM +1;
    fprintf('EM_MixFHMMR try n∞ %d\n',try_EM);
    time = cputime;     
    %%%%%%%%%%%%%%%%%%%
    %  Initialization %
    %%%%%%%%%%%%%%%%%%%
    param = init_MixFHMMR(data, K, R, X, variance_type, ordered_states, init_kmeans, try_EM);

    iter = 0; 
    converge = 0;
    loglik = 0;
    prev_loglik=-inf;
    
    % % EM %%%%
    while ~converge && (iter< max_iter_EM)
        %
        exp_num_trans_ck  = zeros(R,R,n); 
        exp_num_trans_from_l_cg = zeros(R,n);
        %
        exp_num_trans = zeros(R,R,n,K);
        exp_num_trans_from_l = zeros(R,n,K);
        %
        %w_k_fyi = zeros(n,K);
        log_w_k_fyi = zeros(n,K);
        %tau_ik = zeros(n,K);
        %log_tau_ik = zeros(n,K);
        
        %%%%%%%%%%
        % E-Step %
        %%%%%%%%%%
        gamma_ikjr = zeros(n*m,R,K);
        for k=1:K
            % run a hmm for each sequence
            %fkr_xij = zeros(R,m);
            %
            Li = zeros(n,1);% to store the loglik for each example (curve)
            %
            for i=1:n
                %if verbose; fprintf(1,'example %d\n',i); end
                log_fkr_yij = zeros(R,m);                
                Y_i = data(i,:); % ith curve
                for r = 1:R
                    beta_kr = param.beta_kr(:,r,k);
                    if strcmp(variance_type,'common')
                        sigma_kr = param.sigma_k(k);
                        sk = sigma_kr;
                    else
                        sigma_kr = param.sigma_kr(:,k);
                        sk = sigma_kr(r);
                    end
                    z=((Y_i-(phi*beta_kr)').^2)/sk;
                    log_fkr_yij(r,:) = -0.5*ones(1,m).*(log(2*pi)+log(sk)) - 0.5*z;% log pdf yij | c_i = k et z_i = r
                    %fkr_yij(k,:) = normpdf(X_i,(phi*beta_kr)',sqrt(sk));
                end  
                log_fkr_yij  = min(log_fkr_yij,log(realmax));
                log_fkr_yij = max(log_fkr_yij ,log(realmin));
                fkr_yij =  exp(log_fkr_yij);  
                % forwards backwards ( calcul de logProb(Yi)...)
                [gamma_ik, xi_ik, fwd_ik, backw_ik, loglik_i] = forwards_backwards(param.pi_k(:,k), param.A_k(:,:,k), fkr_yij);
                %
                Li(i) = loglik_i; % loglik for the ith curve  ( logProb(Yi))                    
                %
                gamma_ikjr((i-1)*m+1:i*m,:,k) = gamma_ik';%[n*m R K] : "segments" post prob for each cluster k
                %
                exp_num_trans_ck(:,:,i) = sum(xi_ik,3); % [R R n]
                exp_num_trans_from_l_cg(:,i) = gamma_ik(:,1);%[R x n]
                %
            end
            exp_num_trans_from_l(:,:,k) = exp_num_trans_from_l_cg;%[R n K]
            exp_num_trans(:,:,:,k) = exp_num_trans_ck;%[R R n K]
 
            % for computing the global loglik
            %w_k_fyi(:,k) = param.w_k(g)*exp(Li);%[nx1]
            log_w_k_fyi(:,k) = log(param.w_k(k)) + Li;%[nx1]
        end
        log_w_k_fyi = min(log_w_k_fyi,log(realmax));
        log_w_k_fyi = max(log_w_k_fyi,log(realmin)); 
        
        tau_ik = exp(log_w_k_fyi)./(sum(exp(log_w_k_fyi),2)*ones(1,K));%cluster post prob
        
        % % log-likelihood for the n curves
        loglik = sum(log(sum(exp(log_w_k_fyi),2)));
  
        %%%%%%%%%%%
        % M-Step  %
        %%%%%%%%%%%
        
        % Maximization of Q1 w.r.t w_k 
        param.w_k = sum(tau_ik,1)'/n; 
        for k=1:K
            
            if strcmp(variance_type,'common'), s=0; end

            weights_cluster_k = tau_ik(:,k);
            % Maximization of Q2 w.r.t \pi^g
            exp_num_trans_k_from_l =   (ones(R,1)*weights_cluster_k').*exp_num_trans_from_l(:,:,k);%[R x n]
            param.pi_k(:,k) = (1/sum(tau_ik(:,k)))*sum(exp_num_trans_k_from_l,2);% sum over i
            % Maximization of Q3 w.r.t A^g (the trans mat)
            for r=1:R
                if n==1
                    exp_num_trans_k(r,:,:) = (ones(R,1)*weights_cluster_k)'.*squeeze(exp_num_trans(r,:,:,k));
                else
                    %exp_num_trans_k(k,:,:,g)
                    exp_num_trans_k(r,:,:) = (ones(R,1)*weights_cluster_k').*squeeze(exp_num_trans(r,:,:,k));
                end
            end
            if n==1
                temp = exp_num_trans_k;
            else
                temp = sum(exp_num_trans_k,3);%sum over i
            end
            param.A_k(:,:,k) = mk_stochastic(temp);
            % if HMM with order constraints
            if ordered_states
                param.A_k(:,:,k) = mk_stochastic(param.mask.*param.A_k(:,:,k));
            end
            
            % Maximisation de Q4 par rapport aux betak et sigmak
            Ng = sum(tau_ik,1);%nbr of individuals within the cluster k ,k=1...K estimated at iteration q
            %for g=1:K
            ng = Ng(k); %cardinal nbr of the cluster k
            % each sequence i (m observations) is first weighted by the cluster weights
            weights_cluster_k =  repmat((tau_ik(:,k))',m,1);
            weights_cluster_k = weights_cluster_k(:);
            % secondly, the m observations of each sequance are weighted by the
            % wights of each segment k (post prob of the segments for each
            % cluster g)
            gamma_ijk = gamma_ikjr(:,:,k);% [n*m R]           
            
            nm_kr=sum(gamma_ijk,1);% cardinal nbr of the segments r,r=1,...,R within each cluster k, at iteration q              
            
            sigma_kr = zeros(R,1);
            for r=1:R
                nmkr = nm_kr(r);%cardinal nbr of segment r for the cluster k
                weights_seg_k = gamma_ijk(:,r);
                Xkr = (sqrt(weights_cluster_k.*weights_seg_k)*ones(1,p+1)).*X;%[n*m x (p+1)]
                Ykr = (sqrt(weights_cluster_k.*weights_seg_k)).*Y;%[n*m x 1] 
                %  Weighted least squares: maximization w.r.t beta_kr
                beta_kr(:,r) = inv(Xkr'*Xkr )*Xkr'*Ykr; % Maximisation par rapport aux betakr
                % W_kr = diag(weights_cluster_k.*weights_seg_k);
                % beta_kr(:,k) = inv(Phi'*W_kr*Phi)*Phi'*W_kr*X;
                
                % % Maximization w.r.t sigmak :
                z = sqrt(weights_cluster_k.*weights_seg_k).*(Y-X*beta_kr(:,r));
                if strcmp(variance_type,'common')
                    s = s + z'*z;                    
                    ngm = sum(sum((weights_cluster_k*ones(1,R)).*gamma_ijk));
                    sigma_k = s/ngm; 
                else
                    ngmk = sum(weights_cluster_k.*weights_seg_k);
                    sigma_kr(r)=  z'*z/(ngmk);
                end
            end
            param.beta_kr(:,:,k) = beta_kr;
            if strcmp(variance_type,'common')
                param.sigma_k(k) = sigma_k;
            else
                param.sigma_kr(:,k) = sigma_kr;
            end
        end
        iter=iter+1;

        if prev_loglik-loglik > threshold, fprintf(1, 'EM loglik is decreasing from %6.4f to %6.4f!\n', prev_loglik, loglik);end
        if verbose, fprintf(1,'EM_MixFHMMR : Iteration : %d   log-likelihood : %f \n',  iter,loglik);end
           
           converge =  abs((loglik-prev_loglik)/prev_loglik) <= threshold;
           prev_loglik = loglik;           
           stored_loglik(iter) = loglik;
    end % end of EM  loop   
    cputime_total = [cputime_total cputime-time];
    
    MixFHMMR.model = param; 
    if strcmp(variance_type,'common')
       MixFHMMR.stats.paramter_vector = [param.w_k(:); param.A_k(:); param.pi_k(:); param.beta_kr(:); param.sigma_k(:)];      
    else
       MixFHMMR.stats.paramter_vector = [param.w_k(:); param.A_k(:); param.pi_k(:); param.beta_kr(:); param.sigma_kr(:)];      
    end
    MixFHMMR.stats.tau_ik = tau_ik;
    MixFHMMR.stats.gamma_ikjr = gamma_ikjr;
    MixFHMMR.stats.loglik = loglik;
    MixFHMMR.stats.stored_loglik = stored_loglik;    
    MixFHMMR.stats.log_w_k_fyi = log_w_k_fyi;
    
    if MixFHMMR.stats.loglik > best_loglik
       best_loglik = MixFHMMR.stats.loglik;
       best_MixFHMMR = MixFHMMR;
    end
      
    if try_EM>=1,  fprintf('log-lik at convergence: %f \n', MixFHMMR.stats.loglik); end
    
end% Fin de la boucle sur les essais EM

MixFHMMR.stats.loglik = best_loglik;

if try_EM>1,  fprintf('log-lik max: %f \n', MixFHMMR.stats.loglik); end

MixFHMMR = best_MixFHMMR;
% Finding the curve partition by using the MAP rule
[klas, Cig] = MAP(MixFHMMR.stats.tau_ik);% MAP partition of the n sequences
MixFHMMR.stats.klas = klas;

% cas o√π on prend la moyenne des gamma_ijkr

% mean_curves = zeros(m,R,K);
%mean_gamma_ijk = zeros(m,R,K);
smoothed = zeros(m,K);
for k=1:K
    betakr = MixFHMMR.model.beta_kr(:,:,k);
    weighted_segments = sum(MixFHMMR.stats.gamma_ikjr(:,:,k).*(X*betakr),2);
    %
    weighted_segments = reshape(weighted_segments,m,n);
    weighted_clusters = (ones(m,1)*MixFHMMR.stats.tau_ik(:,k)').* weighted_segments;
    smoothed(:,k) = (1/sum(MixFHMMR.stats.tau_ik(:,k)))*sum(weighted_clusters,2);%(1/sum(MixFHMMR.stats.tau_ik(:,k)))*sum(gamma_ikjr(:,:,g).*(X*ones(1,R)),2)
end
MixFHMMR.stats.smoothed = smoothed;
%MixFHMMR.stats.mean_curves = mean_curves;
%MixFHMMR.stats.mean_gamma_ijk = mean_gamma_ijk;
MixFHMMR.stats.cputime = mean(cputime_total);

% optimal sequence for the cluster g
% for g=1:K
% % viterbi
% path_k  = viterbi_path(MixFHMMR.model.pi_k(:,k),MixFHMMR.model.A_k(:,:,g), obslik)
% %     path_k Zjk] = MAP(MixFHMMR.mean_gamma_ijk(:,:,g));%MAP segmentation of each cluster of sequences
%     MixFHMMR.stats.segments(:,k) = path_k;
% end

nu = length(MixFHMMR.stats.paramter_vector);
% BIC AIC et ICL*
MixFHMMR.stats.BIC = MixFHMMR.stats.loglik - (nu*log(n)/2);%n*m/2!
MixFHMMR.stats.AIC = MixFHMMR.stats.loglik - nu;
% ICL*             
% Compute the comp-log-lik 
cig_log_w_k_fyi = (Cig).*(MixFHMMR.stats.log_w_k_fyi);
comp_loglik = sum(sum(cig_log_w_k_fyi,2)); 
MixFHMMR.stats.ICL1 = comp_loglik - nu*log(n)/2;%n*m/2!    
    
    
    
    
    
    
    
    
  

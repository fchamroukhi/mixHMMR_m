function solution =  learn_MixFHMMR_EM(data, G, K, p, ...
    variance_type, ordered_states, total_EM_tries, max_iter_EM, init_kmeans, threshold, verbose)
%   solution =  seq_clust_MixFHMMR(data, G, K, p,fs, variance_type,...
%               order_constraint, total_EM_tries, max_iter_EM, init_kmeans, threshold, verbose)
% Learn a mixture of Hidden Markov Moedel Regression for curve clustering by EM
%
%
% Inputs : 
%
%          1. data :  n curves each curve is composed of m points : dim(Y)=[n m] 
%                * Each curve is observed during the interval [0,T]=[t_1,...,t_m]
%                * t{j}-t_{j-1} = 1/fs (fs: sampling period)    
%          2. G: number of clusters
%          3. K: Number of polynomial regression components (regimes)
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
%          solution : structure containing the following fields:
%                   
%          1. param : a structure containing the model parameters
%                       ({Wg},{alpha_g}, {beta_gk},{sigma_gk}) for g=1,...,G and k=1...K. 
%              1.1 Wg = (Wg1,...,w_gK-1) parameters of the logistic process:
%                  matrix of dimension [(q+1)x(K-1)] with q the order of logistic regression.
%              1.2 beta_g = (beta_g1,...,beta_gK) polynomial regression coefficient vectors: matrix of
%                  dimension [(p+1)xK] p being the polynomial  degree.
%              1.3 sigma_g = (sigma_g1,...,sigma_gK) : the variances for the K regmies. vector of dimension [Kx1]
%              1.4 pi_jgk :logistic proportions for cluster g
%
%          2. Psi: parameter vector of the model: Psi=({Wg},{alpha_g},{beta_gk},{sigma_gk}) 
%                  column vector of dim [nu x 1] with nu = nbr of free parametres
%          3. h_ig = prob(curve|cluster_g) : post prob (fuzzy segmentation matrix of dim [nxG])
%          4. c_ig : Hard partition obtained by the AP rule :  c_{ig} = 1
%                    if and only c_i = arg max_g h_ig (g=1,...,G)
%          5. klas : column vector of cluster labels
%          6. tau_ijgk prob(y_{ij}|kth_segment,cluster_g), fuzzy
%          segmentation for the cluster g. matrix of dimension
%          [nmxK] for each g  (g=1,...,G).
%          7. Ex_g: curve expectation: sum of the polynomial components beta_gk ri weighted by 
%             the logitic probabilities pij_gk: Ex_g(j) = sum_{k=1}^K pi_jgk beta_gk rj, j=1,...,m. Ex_g 
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

[n, m] = size(data);%n  nbre de signaux (individus); m: nbre de points pour chaque signal
% % construction des matrices de regression
t=0:m-1;
%t= linspace(0,1,m);
[phi] = designmatrix(t,p);% pour 1 courbe
Phi = repmat(phi,n,1);%pour les n courbes
%
Y=reshape(data',[],1); 


% % main algorithm
try_EM = 0; 
best_loglik = -inf;
cputime_total = [];
while (try_EM < total_EM_tries)
    try_EM = try_EM +1;
    fprintf('EM try n° %d\n',try_EM);
    time = cputime;     
    %%%%%%%%%%%%%%%%%%%
    %  Initialization %
    %%%%%%%%%%%%%%%%%%%
    param = initialize_MixFHMMR(data, G, K, Phi, variance_type, ordered_states, init_kmeans, try_EM);

    iter = 0; 
    converge = 0;
    loglik = 0;
    prev_loglik=-inf;
    
    % % EM %%%%
    while ~converge && (iter< max_iter_EM)
        %
        exp_num_trans_cg  = zeros(K,K,n); 
        exp_num_trans_from_l_cg = zeros(K,n);
        %
        exp_num_trans = zeros(K,K,n,G);
        exp_num_trans_from_l = zeros(K,n,G);
        %
        %w_g_Pxi = zeros(n,G);
        log_w_g_Pxi = zeros(n,G);
        %tau_ig = zeros(n,G);
        %log_tau_ig = zeros(n,G);
        
        %%%%%%%%%%
        % E-Step %
        %%%%%%%%%%
        gamma_igjk = zeros(n*m,K,G);
        for g=1:G
            % run a hmm for each sequence
            %fgk_xij = zeros(K,m);
            %
            Li = zeros(n,1);% to store the loglik for each example (curve)
            %
            for i=1:n
                %if verbose; fprintf(1,'example %d\n',i); end
                log_fgk_yij = zeros(K,m);                
                Y_i = data(i,:); % ith curve
                for k = 1:K
                    beta_gk = param.beta_gk(:,k,g);
                    if strcmp(variance_type,'common')
                        sigma_gk = param.sigma_g(g);
                        sk = sigma_gk;
                    else
                        sigma_gk = param.sigma_gk(:,g);
                        sk = sigma_gk(k);
                    end
                    z=((Y_i-(phi*beta_gk)').^2)/sk;
                    log_fgk_yij(k,:) = -0.5*ones(1,m).*(log(2*pi)+log(sk)) - 0.5*z;% log pdf cond à c_i = g et z_i = k de xij
                    %fgk_xij(k,:) = normpdf(X_i,(phi*beta_gk)',sqrt(sk));
                end  
                log_fgk_yij  = min(log_fgk_yij,log(realmax));
                log_fgk_yij = max(log_fgk_yij ,log(realmin));
                fgk_xij =  exp(log_fgk_yij);  
                % forwards backwards ( calcul de logProb(Xi)...)
                [gamma_ig, xi_ig, fwd_ig, backw_ig, loglik_i] = forwards_backwards(param.pi_g(:,g), param.A_g(:,:,g), fgk_xij);
                %
                Li(i) = loglik_i; % loglik for the ith curve  ( logProb(Xi))                    
                %
                gamma_igjk((i-1)*m+1:i*m,:,g) = gamma_ig';%[n*m K G] : "segments" post prob for each cluster g
                %
                exp_num_trans_cg(:,:,i) = sum(xi_ig,3); % [K K n]
                exp_num_trans_from_l_cg(:,i) = gamma_ig(:,1);%[K x n]
                %
            end
            exp_num_trans_from_l(:,:,g) = exp_num_trans_from_l_cg;%[K n G]
            exp_num_trans(:,:,:,g) = exp_num_trans_cg;%[K K n G]
 
            % for computing the global loglik
            %w_g_Pxi(:,g) = param.w_g(g)*exp(Li);%[nx1]
            log_w_g_Pxi(:,g) = log(param.w_g(g)) + Li;%[nx1]
        end
        log_w_g_Pxi = min(log_w_g_Pxi,log(realmax));
        log_w_g_Pxi = max(log_w_g_Pxi,log(realmin)); 
        
        tau_ig = exp(log_w_g_Pxi)./(sum(exp(log_w_g_Pxi),2)*ones(1,G));%cluster post prob
        
        % % log-likelihood for the n curves
        loglik = sum(log(sum(exp(log_w_g_Pxi),2)));
  
        %%%%%%%%%%%
        % M-Step  %
        %%%%%%%%%%%
        
        % Maximization of Q1 w.r.t w_g 
        param.w_g = sum(tau_ig,1)'/n; 
        for g=1:G
            
            if strcmp(variance_type,'common'), s=0; end

            weights_cluster_g = tau_ig(:,g);
            % Maximization of Q2 w.r.t \pi^g
            exp_num_trans_g_from_l =   (ones(K,1)*weights_cluster_g').*exp_num_trans_from_l(:,:,g);%[K x n]
            param.pi_g(:,g) = (1/sum(tau_ig(:,g)))*sum(exp_num_trans_g_from_l,2);% sum over i
            % Maximization of Q3 w.r.t A^g (the trans mat)
            for k=1:K
                if n==1
                    exp_num_trans_g(k,:,:) = (ones(K,1)*weights_cluster_g)'.*squeeze(exp_num_trans(k,:,:,g));
                else
                    %exp_num_trans_g(k,:,:,g)
                    exp_num_trans_g(k,:,:) = (ones(K,1)*weights_cluster_g').*squeeze(exp_num_trans(k,:,:,g));
                end
            end
            if n==1
                temp = exp_num_trans_g;
            else
                temp = sum(exp_num_trans_g,3);%sum over i
            end
            
            param.A_g(:,:,g) = mk_stochastic(temp);
            % if HMM with order constraints
            if ordered_states
                param.A_g(:,:,g) = mk_stochastic(param.mask.*param.A_g(:,:,g));
            end
            
            % Maximisation de Q4 par rapport aux betak et sigmak
            Ng = sum(tau_ig,1);%nbr of individuals within the cluster g ,g=1...G estimated at iteration q
            %for g=1:G
            ng = Ng(g); %cardinal nbr of the cluster g
            % each sequence i (m observations) is first weighted by the cluster weights
            weights_cluster_g =  repmat((tau_ig(:,g))',m,1);
            weights_cluster_g = weights_cluster_g(:);
            % secondly, the m observations of each sequance are weighted by the
            % wights of each segment k (post prob of the segments for each
            % cluster g)
            gamma_ijk = gamma_igjk(:,:,g);% [n*m K]           
            
            nm_gk=sum(gamma_ijk,1);% cardinal nbr of the segments k,k=1,...,K within each cluster g, at iteration q              
            
            sigma_gk = zeros(K,1);
            for k=1:K
                nmgk = nm_gk(k);%cardinal nbr of segment k for the cluster g
                weights_seg_k = gamma_ijk(:,k);
                phigk = (sqrt(weights_cluster_g.*weights_seg_k)*ones(1,p+1)).*Phi;%[n*m x (p+1)]
                Xgk = (sqrt(weights_cluster_g.*weights_seg_k)).*Y;%[n*m x 1] 
                %  Weighted least squares: maximization w.r.t beta_gk
                beta_gk(:,k) = inv(phigk'*phigk )*phigk'*Xgk; % Maximisation par rapport aux betak
                % W_gk = diag(weights_cluster_g.*weights_seg_k);
                % beta_gk(:,k) = inv(Phi'*W_gk*Phi)*Phi'*W_gk*X;
                
                % % Maximization w.r.t sigmak :
                z = sqrt(weights_cluster_g.*weights_seg_k).*(Y-Phi*beta_gk(:,k));
                if strcmp(variance_type,'common')
                    s = s + z'*z;                    
                    ngm = sum(sum((weights_cluster_g*ones(1,K)).*gamma_ijk));
                    sigma_g = s/ngm; 
                else
                    ngmk = sum(weights_cluster_g.*weights_seg_k);
                    sigma_gk(k)=  z'*z/(ngmk);
                end
            end
            param.beta_gk(:,:,g) = beta_gk;
            if strcmp(variance_type,'common')
                param.sigma_g(g) = sigma_g;
            else
                param.sigma_gk(:,g) = sigma_gk;
            end
        end
        iter=iter+1;

        if prev_loglik-loglik > threshold, fprintf(1, 'EM loglik is decreasing from %6.4f to %6.4f!\n', prev_loglik, loglik);end
        if verbose, fprintf(1,'EM : Iteration : %d   log-likelihood : %f \n',  iter,loglik);end
           
           converge =  abs((loglik-prev_loglik)/prev_loglik) <= threshold;
           prev_loglik = loglik;           
           stored_loglik(iter) = loglik;
    end % end of EM  loop   
    cputime_total = [cputime_total cputime-time];
    
    solution.param = param; 
    if strcmp(variance_type,'common')
       solution.Psi = [param.w_g(:); param.A_g(:); param.pi_g(:); param.beta_gk(:); param.sigma_g(:)];      
    else
       solution.Psi = [param.w_g(:); param.A_g(:); param.pi_g(:); param.beta_gk(:); param.sigma_gk(:)];      
    end
    solution.tau_ig = tau_ig;
    solution.gamma_igjk = gamma_igjk;
    solution.loglik = loglik;
    solution.stored_loglik = stored_loglik;    
    solution.log_w_g_Pxi = log_w_g_Pxi;
    
    if solution.loglik > best_loglik
       best_loglik = solution.loglik;
       best_solution = solution;
    end
      
    if try_EM>=1,  fprintf('log-lik at convergence: %f \n', solution.loglik); end
    
end% Fin de la boucle sur les essais EM

solution.loglik = best_loglik;

if try_EM>1,  fprintf('log-lik max: %f \n', solution.loglik); end

solution = best_solution;
% Finding the curve partition by using the MAP rule
[klas, Cig] = MAP(solution.tau_ig);% MAP partition of the n sequences
solution.klas = klas;

% cas où on prend la moyenne des gamma_ijgk

% mean_curves = zeros(m,K,G);
%mean_gamma_ijk = zeros(m,K,G);
smoothed = zeros(m,G);
for g=1:G
    betagk = solution.param.beta_gk(:,:,g);
    weighted_segments = sum(solution.gamma_igjk(:,:,g).*(Phi*betagk),2);
    %
    weighted_segments = reshape(weighted_segments,m,n);
    weighted_clusters = (ones(m,1)*solution.tau_ig(:,g)').* weighted_segments;
    smoothed(:,g) = (1/sum(solution.tau_ig(:,g)))*sum(weighted_clusters,2);%(1/sum(solution.tau_ig(:,g)))*sum(gamma_igjk(:,:,g).*(X*ones(1,K)),2)
end
solution.smoothed = smoothed;
%solution.mean_curves = mean_curves;
%solution.mean_gamma_ijk = mean_gamma_ijk;
solution.cputime = mean(cputime_total);

% optimal sequence for the cluster g
% for g=1:G
% % viterbi
% path_g  = viterbi_path(solution.param.pi_g(:,g),solution.param.A_g(:,:,g), obslik)
% %     path_g Zjk] = MAP(solution.mean_gamma_ijk(:,:,g));%MAP segmentation of each cluster of sequences
%     solution.segments(:,g) = path_g;
% end

nu = length(solution.Psi);
% BIC AIC et ICL*
solution.BIC = solution.loglik - (nu*log(n)/2);%n*m/2!
solution.AIC = solution.loglik - nu;
% ICL*             
% Compute the comp-log-lik 
cig_log_w_g_Pxi = (Cig).*(solution.log_w_g_Pxi);
comp_loglik = sum(sum(cig_log_w_g_Pxi,2)); 
solution.ICL1 = comp_loglik - nu*log(n)/2;%n*m/2!    
    
    
    
    
    
    
    
    
  

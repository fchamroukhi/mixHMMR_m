function param = init_MixFHMMR(data, K, R, Phi, variance_type, ordered_states, init_kmeans, try_algo)
%
%
%
%%%%%%%%%%%%%%%%%%%%%%%% FC %%%%%%%%%%%%%%%%%%%
D = data;
[n, m]=size(D);

% % 1. Initialization of cluster weights
param.w_k=1/K*ones(K,1);
%Initialization of the model parameters for each cluster
if init_kmeans
    max_iter_kmeans = 400;
    n_tries_kmeans = 20;
    verbose_kmeans = 0;
    res_kmeans = myKmeans(D, K, n_tries_kmeans, max_iter_kmeans, verbose_kmeans);
    for k=1:K
        Yk = D(res_kmeans.klas==k ,:); %if kmeans
        param_init =  init_hmm_regression(Yk, R, Phi, ordered_states, variance_type, try_algo);
        
        % 3. Initialisation de la matrice des transitions        
        param.A_k(:,:,k)  =  param_init.trans_mat;
        if ordered_states
            param.mask = param_init.mask;
        end
        % 2. Initialisation de \pi_k
        param.pi_k(:,k) = param_init.initial_prob;%[1;zeros(R-1,1)];               
        % 4. Initialisation des coeffecients de regression et des variances.
        param.beta_kr(:,:,k) = param_init.betar;
        if strcmp(variance_type,'common')
            param.sigma_k(k) = param_init.sigma;
        else
            param.sigma_kr(:,k) = param_init.sigmar;
        end
    end
else
    ind = randperm(n);
    for k=1:K
        if k<K
            Yk = D(ind((k-1)*round(n/K) +1 : k*round(n/K)),:);
        else
            Yk = D(ind((k-1)*round(n/K) +1 : end),:);
        end
        param_init =  init_hmm_regression(Yk, R, Phi, ordered_states, variance_type, try_algo);
        % 3. Initialisation de la matrice des transitions        
        param.A_k(:,:,k)  =  param_init.trans_mat;
        if ordered_states
            param.mask = param_init.mask;
        end
        % 2. Initialisation de \pi^g
        param.pi_k(:,k) = param_init.initial_prob;%[1;zeros(K-1,1)];               
        % 4. Initialisation des coeffecients de regression et des variances.
        param.beta_kr(:,:,k) = param_init.betar;
        if strcmp(variance_type,'common')
            param.sigma_k(k) = param_init.sigma;
        else
            param.sigma_kr(:,k) = param_init.sigmar;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param =  init_hmm_regression(data, R, Phi, ordered_states, variance_type, try_algo)
% init_hmm_regression estime les paramètres initiaux d'un modèle de regression
% à processus markovien cache où la loi conditionnelle des observations est une gaussienne
%
% Entrees :
%       
%        data  = n sequences each sequence is of m points 
%        signaux les observations sont monodimentionnelles)
%        R : nbre d'états (classes) cachés
%        X :  matrice de r�gression
%
% Sorties :
%
%         param : parametres initiaux du modele. structure
%         contenant les champs: para: structrure with the fields:
%         * le HMM initial          
%         1. initial_prob (k) = Pr(Z(1) = k) avec k=1,...,K. loi initiale de z.
%         2. trans_mat(\ell,k) = Pr(z(i)=k | z(i-1)=\ell) : matrice des transitions
%         *         
%         3.betar : le vecteur parametre de regression associe a la classe k.
%         vecteur colonne de dim [(p+1)x1]
%         4. sigmar(k) = variance de x(i) sachant z(i)=k; sigmar(j) =
%         sigma^2_k.
% 
% Faicel Chamroukhi, Novembre 2008
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% 1. Initialization of the HMM parameters
if ordered_states
    % Initialisation en tenant compte de la contrainte:
    % Initialisation de la matrice des transitions
    mask = eye(R);%mask d'ordre 1
    % mask = eye(K).*rand(K,K);%initialisation al�atoire
    for k=1:R-1
        ind = find(mask(k,:) ~= 0);
        mask(k,ind+1) = 1;
    end
    % Initialisation de la loi initiale de la variable cachee
    param.initial_prob = [1;zeros(R-1,1)];
    param.trans_mat = normalize(mask,2);%
    param.mask = mask;
 else
    % Initialisation de la loi initiale de la variable cachee
    param.initial_prob = [1;zeros(R-1,1)];%1/K*ones(K,1);
    param.trans_mat = mk_stochastic(rand(R));
end


% % 2.  Initialisation of regression coefficients and variances

regression_param = init_regression_param(data, R, Phi, variance_type, try_algo);

param.betar = regression_param.betar;
if strcmp(variance_type,'common')
    param.sigma = regression_param.sigma;
else
    param.sigmar = regression_param.sigmar;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function param = init_regression_param(data, R, X,variance_type, try_algo)
 
% init_regression_param initialize the Regresssion model with Hidden Logistic Process
%
% X: regression matrix
%
%%%%%%%%%%%%%%%%%%%% Faicel Chamroukhi %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n, m] = size(data);
% p = size(phi,2)-1;
Y = data;

if strcmp(variance_type,'common')
    s=0;
end 
%

if try_algo ==1
    %decoupage de l'echantillon (signal) en K segments
    zi = round(m/R)-1;
    for r=1:R
        i = (r-1)*zi+1;
        j = r*zi;
        
        Yij = Y(:,i:j);
        Yij = reshape(Yij',[],1);
        
        X_ij=repmat(X(i:j,:),n,1);
           
        br = inv(X_ij'*X_ij)*X_ij'*Yij; 
        
        param.betar(:,r) = br;
        
        if strcmp(variance_type,'common')
            s=s+ sum((Yij-X_ij*br).^2);
           param.sigma = s/(n*m);
         else
             mk = j-i+1 ; 
             z = Yij-X_ij*br;
             sk = z'*z/(n*mk); 
             param.sigmar(r) = sk;
         end
     end
 else % initialisation aléatoire
     Lmin= round(m/(R+1));%nbr pts min dans un segments
     t_r_init = zeros(1,R+1);
     t_r_init(1) = 0;         
     R_1=R;
     for r = 2:R
         R_1 = R_1-1;
         temp = t_r_init(r-1)+Lmin:m-R_1*Lmin;
         ind = randperm(length(temp));
         t_r_init(r)= temp(ind(1));                      
     end
     t_r_init(R+1) = m;
     %model.tk_init = tk_init;
     for r=1:R
         i = t_r_init(r)+1;
         j = t_r_init(r+1);
         Yij = Y(:,i:j);
         Yij = reshape(Yij',[],1);
        
        X_ij=repmat(X(i:j,:),n,1);
           
        br = inv(X_ij'*X_ij)*X_ij'*Yij; 
        param.betar(:,r) = br;
        
        if strcmp(variance_type,'common')
           s=s+ sum((Yij-X_ij*br).^2);
           param.sigma = s/(n*m);
        else
             mk = j-i+1 ;%length(Yij);
             z = Yij-X_ij*br;
             sk = z'*z/(n*mk); 
             param.sigmar(r) = sk;
         end
     end
 end
    
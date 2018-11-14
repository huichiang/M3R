# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import operator as op
from functools import reduce
import seaborn as sns
sns.set(color_codes=True)
import scipy

def bintrials(nexp, N, n1, nt):
    #Negative binomial method as described in Section 5.2
    
    #Inputs
    #nexp: Number of times to run the experiment
    #N: Total number of fish in the pond
    #n1: Number of fish caught on the first day
    #nt: Number of tagged fish caught on the second day

    #Output
    #ntrials: nexp negative binomial trials, each with number of untagged fish

    pn = n1/N #probability of catching a tagged fish
    ntrials = np.zeros(nexp)

    for i in range(nexp):
        ntrials[i] = np.random.negative_binomial(nt,pn)

    return ntrials

def fishtrials(nexp, N, n1, nt):
    #Inputs
    #nexp: Number of times to run the experiment
    #N: Total number of fish in the pond
    #n1: Number of fish caught on the first day
    #nt: Number of tagged fish caught on the second day

    #Output
    #ntrials: nexp binomial trials

    assert(N<n1), "Cannot catch more fish than total fish"
    assert(n1<nt), "Cannot catch more tagged fish than present"

    ntrials = np.zeros(nexp)
    remaining = [N,n1]
    caught = [0,0]

    for i in range(nexp):
        remaining = [N,n1]
        caught = [0,0]
        while caught[1] != nt:
            if np.random.binomial(1,remaining[1]/remaining[0]) == 1:
                caught[0] += 1
                caught[1] += 1
                remaining[0] -= 1
                remaining[1] -= 1
            else:
                caught[0] += 1
                remaining[0] -= 1
        ntrials[i] = caught[0]

    return ntrials

def ptrials(N,pcatch,ncatch1,ncatch2,ntrials):
    #Method 1 as described in Section 5.1
    
    #Inputs
    #N: Total number of fish in the pond
    #pcatch: Probability of catching a fish with each ncatch
    #ncatch1: Number of times to try catching a fish the first day
    #ncatch2: Number of times to try catching a fish the second day
    #ntrials: Number of times to run the experiment

    #Outputs
    #n1: Fish tagged on the first day
    #n2: Total fish caught on the second day
    #nt: Tagged fish caught on the second day

    n1 = np.zeros(ntrials)
    n2 = np.zeros(ntrials)
    nt = np.zeros(ntrials)

    for j in range(ntrials):
        n1[j] = np.random.binomial(ncatch1,pcatch)
        remaining = [N,n1[j]]
        for i in range(ncatch2):
            if np.random.binomial(1,pcatch) == 1:
                if np.random.binomial(1,remaining[1]/remaining[0]) == 1:
                    n2[j] = n2[j] + 1
                    nt[j] = nt[j] + 1
                    remaining[0] -= 1
                    remaining[1] -= 1
                else:
                    n2[j] = n2[j] + 1
                    remaining[0] -= 1

    return n1,n2,nt

def ncr(n, r):
    #'Choose' function ie computes n!/(n-r)!r!
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    
    return numer//denom

def nhypergeom(N, K, r):
    #Creates a list of probabilities following of an r.v. following a negative
    #hypergeometric distribution

    #Inputs
    #N: Total number of objects
    #K: Total number of 'success' elements (untagged fish)
    #r: Total number of 'failure' elements to stop the experiment at
        #(number of tagged fish to catch)

    #Output
    #k: Vector of probabilities over the support of K

    k = np.zeros(K+1) #Support of K

    for i in range(K+1):
        k[i] =  ncr(i+r-1,i)*ncr(N-r-i,K-i)/ncr(N,K)

    return k

def htrials(nexp, N, n1, nt):
    #Negative hypergeometric method as in Section 5.2
    
    #Inputs
    #nexp: Number of times to run the experiment
    #N: Total number of fish in the pond
    #n1: Total number of tagged fish in the pond
    #nt: Number of tagged fish caught to stop experiment

    #Output
    #ntrials: Nexp trials of untagged fish caught

    ntrials = np.zeros(nexp)
    probs = nhypergeom(N,N-n1,nt) #probabilities of each k
    support = np.arange(len(probs)) #support of K

    for i in range(nexp):
        #Sample from the negative hypergeometric distribution
        ntrials[i] = np.random.choice(support,p=probs)

    #plt.hist(ntrials)
    #plt.show()

    return ntrials

def Ntrials(nexp, n1, n2, nt):
    #Sampling from the posterior distribution of N
    
    #Inputs
    #nexp: Number of times to run the experiment
    #n1: Total number of tagged fish in the pond
    #n2: Total number of fish caught on second day
    #nt: Number of tagged fish caught to stop experiment

    #Output
    #ntrials: nexp trials

    ntrials = np.zeros(nexp)
    support = np.arange(n1+n2-nt,2500) #support of N
    probs = np.zeros(len(support)) #probability vector

    for i in range(len(support)):
        num = ncr(support[i]-n2,support[i]-n1-n2+nt)
        den = ncr(support[i],support[i]-n1)
        probs[i] = num/den

    probs = probs/sum(probs)
    for i in range(nexp):
        #Sample from the negative hypergeometric distribution
        ntrials[i] = np.random.choice(support,p=probs)

    return ntrials

def gibbs(N, n1, n2, nt, t, iters):
    #Gibbs sampler with t number of fish with unknown tags
    
    #Input:
    #N: Total number of fish in the pond
    #n1: Total number of tagged fish in the pond
    #n2: Total number of fish caught on second day
    #nt: Number of tagged fish caught to stop the experiment
    #t: Vector of fish with missing fins
    #iters: Number of iterations

    #Output:
    #nsample: Sample of total number of fish in the pond 
                #(Can also return number of untagged fish caught)

    nsample = np.zeros(int(iters/20))

    count = 0
    fill = 0
    while nsample[-1] == 0:
        count += 1
        print(count)
        t = t.astype(int)
        #Sample the fish with missing fins
        for i in range(len(t)):
            t[i] = np.random.binomial(1, ((n1-nt-sum(t))/(N-n2-1)))

        #Sample the total number of fish on the second day, n2
        nhgprobs = nhypergeom(N,N-n1,nt+sum(t))
        nhgsupport = np.arange(len(nhgprobs))
        untagged = np.random.choice(nhgsupport,p=nhgprobs)+ 5 - sum(t)
        n2 = untagged + nt + sum(t)

        #Sample the total number of fish in the pond
        Nsupport = np.arange(n1+n2-nt,2501,dtype=int)
        Nprobs = np.zeros(len(Nsupport))
        for i in range(len(Nsupport)):
            Nprobs[i] = ncr(Nsupport[i]-n2,Nsupport[i]-n1-n2+nt+sum(t))/ncr(Nsupport[i],Nsupport[i]-n1)
        N = np.random.choice(Nsupport, p=Nprobs/sum(Nprobs))
        if count % 5 == 0 :
            if n2 == 200:
                nsample[fill] = N
                fill += 1
                print('fill',fill)

    return nsample
    
def inv_chisq(n, df, wsq):
    #Sample of size n from an inverse chi squared distribution with df degrees
    #of freedom and scale parameter wsq

    #Input:
    #n: Number of samples
    #df: Degrees of freedom
    #wsq: Scale parameter

    #Output:
    #nsample: n sized vector
    
    #First sample from a chi squared distribution with df degrees of freedom
    temp = np.random.chisquare(df, n)
    
    #Calculate the scaled inverse
    sample = float(df)*float(wsq)/temp
    
    return sample
    
def kelly(k, sigmax, sigmay, sigmaxy, x, y, etai, censor):
    #Kelly Gibbs sampler in Section 6.2
    
    #Input:
    #k: Number of Gaussian functions for the mixture model
    #sigmax: n sized vector containing variance of each x
    #sigmay: n sized vector containing variance of each y
    #sigmaxy: n sized vector containing covariance between each x,y pair
    #x: n sized vector containing x
    #y: n sized vector containing y
    #etai: n sized vector containing estimate of etai
    #censor: Values below which are censored
    
    #Output:
    #Note that the following outputs were picked specifically for plotting
    #purposes for the report. Samples of any other parameter can also be picked
    #postalpha: Sample for alpha
    #postbeta: Sample for beta
    #postsigma: Sample for intrinsic variance sigma
    #postcensoreta: Sample for eta of a single censored point
    #postcensorxi: Sample for xi of a single censored point
    #posteta: Sample for eta of a single uncensored point
    #postxi: Sample for xi of a single uncensored point
    #uncensoredlist: Indicies of uncensored points
    #censorlist: Indicies of censored points
    #measurexc: Observed x for selected censored point
    #measurexu: Observed x for selected uncensored point
    #measureyu: Observed y for selected uncensored point
    #etai: Single sample for full eta
    #xi: Single sample for full xi
        
    n = len(sigmax)
    #Autocorrelation for each x,y pair
    rhoxy = np.divide(sigmaxy, np.sqrt(np.multiply(sigmax, sigmay)))
    
    #Initialise output
    postalpha = np.zeros(200)
    postbeta = np.zeros(200)
    postsigma = np.zeros(200)
    postcensoreta = np.zeros(200)
    postcensorxi = np.zeros(200)
    posteta = np.zeros(200)
    postxi = np.zeros(200)
    
    #Parameter values drawn from prior densities:
        #wsq: Scale hyperparameter of prior for tausq
        #mu0: Mean hyperparameter of means of Gaussian functions
        #tausq: k sized vector containing variance of k Gaussian functions
        #usq: Variance hyperparameter of means of Gaussian functions
        #muk: k sized vector containing mean of k Gaussian functions
        #pi: k sized vector containing the probability of drawing data from the 
            #kth Gaussian function
        #sigma: Variance of regression
        #alpha: Regression intercept term
        #beta: Regression covariate
        #G: n*k sized matrix containing class membership of the mixture model
    
    #Draw initial values based on prior distributions
    wsq = np.random.uniform(0,1) 
    mu0 = np.random.uniform(-1,1)
    tausq = inv_chisq(k,n/2,wsq)
    usq = inv_chisq(1,1,np.sqrt(wsq)) 
    muk = np.random.normal(mu0,np.sqrt(usq),k)
    pi = np.random.dirichlet(np.ones(k))
    sigma = np.random.uniform(0,2)
    alpha,beta = np.random.uniform(-1,1,2)
    G = np.zeros(np.array([n,k]))
    for i in range(n):
        G[i,:] = np.random.multinomial(1,pi)
    
    #List of censored and uncensored data points
    censorlist = []
    uncensoredlist = []
    for i in range(n):
       if y[i] == censor :
           censorlist.append(i)
       else:
           uncensoredlist.append(i)
    count = 0
    fill = 0
    
    #Picks a single point from the censored and uncensored list
    measurexc = np.copy(x[censorlist[0]])
    measurexu = np.copy(x[uncensoredlist[0]])
    measureyu = np.copy(y[uncensoredlist[0]])
        
    while postalpha[-1] == 0:
        #Censoring
        for vals in censorlist:
            #Simulate new value for censored y
            y[vals] = np.random.normal(etai[vals],np.sqrt(sigmay[vals]))
            #Only accept if value is less than threshold
            while y[vals] >= censor:
                y[vals] = np.random.normal(etai[vals],np.sqrt(sigmay[vals]))
            if count == 0:
                print(vals)
        
        #Equation 55
        col = np.reciprocal(np.multiply(sigmax,(1-np.square(rhoxy))))+(beta**2/sigma)
        sigmaxik = np.tile(col,(k,1))
        sigmaxik = sigmaxik.transpose()
        sigmaxik = np.reciprocal(np.tile(1/tausq,(n,1)) + sigmaxik)
        print('Equation 58')
        print('sigmaxik',sigmaxik)
    
        #Equation 57
        sigmaxi = np.multiply(G,sigmaxik)
        sigmaxi = sigmaxi.sum(axis=1)
        print('Equation 57')
        print('sigmaxi',sigmaxi)    
        
        #Equation 56
        xixyhat = x + (sigmaxy/sigmay)*(etai-y)
        print('Equation 56')
        print('xixyhat',xixyhat)
    
        #Equation 55
        xiikhat = np.divide(xixyhat,np.multiply(sigmax, (1-np.square(rhoxy))))
        xiikhat = xiikhat + ((1/beta)*(etai-alpha)/sigma) 
        xiikhat = np.tile(xiikhat, (k,1))
        xiikhat = xiikhat.transpose()
        temp = np.tile(muk/tausq, (n,1))
        temp2 = np.tile(sigmaxi, (k,1)).transpose()
        xiikhat = np.multiply((xiikhat + temp),temp2)
        print('Equation 55')
        print('xiikhat',xiikhat)        
        
        #Equation 54
        xihat = np.multiply(G,xiikhat)
        xihat = xihat.sum(axis=1)
        print('Equation 54')
        print('xihat',xihat)
        
        #Equation 53
        xi = np.random.normal(xihat, np.sqrt(sigmaxi))
        print('Equation 53')
        print('xi',xi)
        
        #Equation 68
        first = 1/(sigmay*(1-np.square(rhoxy)))
        second = 1/sigma
        sigmaetai = 1/(first+second)
        print('sigmaetai',sigmaetai)
        
        #Equation 67
        tempnum = y + sigmaxy*(xi-x)/sigmax
        tempden = sigmay*(1-np.square(rhoxy))
        etaihat = sigmaetai*(tempnum/tempden + (alpha + beta*xi)/sigma)
        print('Equation 67')
        print('etaihat',etaihat)
        
        #Equation 66
        etai = np.random.normal(etaihat, np.sqrt(sigmaetai))
        print('Equation 66')
        print('etai',etai)
        
        #Equation 73 and 74
        for i in range(n):
            qk = scipy.stats.norm.pdf(xi[i],muk,np.sqrt(tausq))   
            qk = qk*pi
            qk /= sum(qk)
            G[i,:] = np.random.multinomial(1,qk)
        print('Equation 74')
                
        #Equation 76
        X = np.transpose(np.vstack((np.ones(len(xi)),xi)))
        XtXinv = np.linalg.inv(np.matmul(np.transpose(X),X))
        Xteta = np.matmul(np.transpose(X),etai)
        chat = np.matmul(XtXinv, Xteta)
        print('Equation 76')
        print('chat',chat)
        
        #Equation 77
        sigmachat = XtXinv*sigma
        print('Equation 77')
        print('sigmachat',sigmachat)
        
        #Equation 75
        alpha, beta = np.random.multivariate_normal(chat,sigmachat)
        print('Equation 75')
        print('alpha',alpha)
        print('beta',beta)
        
        #Equation 80
        ssq = sum(np.square(etai-alpha-beta*xi))/(n-2)
        print('Equation 80')
        print('ssq',ssq)
        print('ssum',sum(np.square(etai-alpha-beta*xi)))
        
        #Equation 78
        sigma = (inv_chisq(1,n-2,ssq))
        print('Equation 78')
        print('sigma',sigma)
        
        #Equation 82
        nk = G.sum(axis=0)
        print('Equation 82')
        print('nk',nk)
                
        #Equation 81
        pi = np.random.dirichlet(nk+1)
        print('Equation 81')
        print('pi',pi)
        
        #Equation 86
        sigmamuhatk = 1/(1/usq + nk/tausq)
        print('Equation 86')
        print('sigmamuhatk',sigmamuhatk)
        
        #Equation 85
        xikbar = np.zeros(k)
        temp = np.matmul(np.transpose(G),xi)
        for i in range(k):
            #Prevent dividing by 0
            if nk[i] == 0:
                xikbar[i] = 0
            else:
                xikbar[i] = temp[i]/nk[i] 
        print('Equation 85')
        print('xikbar',xikbar)
        
        #Equation 84
        right = mu0/usq + nk*xikbar/tausq
        mukhat = sigmamuhatk*right
        print('Equation 84')
        print('mukhat',mukhat)
        
        #Equation 83
        muk = np.random.normal(mukhat,np.sqrt(sigmamuhatk))
        print('Equation 83')
        print('muk',muk)
        
        #Equation 89
        tksq = np.zeros(k)
        for j in range(k):
            temp1 = np.square(xi-muk[j])
            temp2 = np.matmul(np.transpose(G[:,j]),temp1)
            tksq[j] = (temp2+wsq)/(nk[j]+1)
        print('Equation 89')
        print('tksq',tksq)
        
        #Equation 87
        for i in range(k):
            tausq[i] = inv_chisq(1,nk[i]+1.0,tksq[i])
        print('Equation 87')
        print('tausq',tausq)
        
        #Equation 94
        mubar = np.mean(muk)
        print('Equation 94')
        print('mubar',mubar)
        
        #Equation 93
        mu0 = np.random.normal(mubar, np.sqrt(usq/k))
        print('Equation 93')
        print('mu0',mu0)
        
        #Equation 97
        usqhat = (sum(np.square(muk-mu0))+wsq)/(k+1)
        print('Equation 97')
        print('usqhat',usqhat)
        
        #Equation 95
        usq = inv_chisq(1,k+1,usqhat)
        print('Equation 95')
        print('usq',usq)

        #Equation 103,101
        b = (1/usq+sum(1/tausq))/2
        wsq = np.random.gamma((k+3)/2,1/b)
        print('Equation 103')
        print('wsq',wsq)
        
        #Burn first 1000 samples, and thin by recording every 5th sample after
        if count >= 1000 and count % 5 == 0:
            postalpha[fill] = alpha
            postbeta[fill] = beta
            postsigma[fill] = sigma
            postcensoreta[fill] = etai[censorlist[0]]
            postcensorxi[fill] = xi[censorlist[0]]
            posteta[fill] = etai[uncensoredlist[0]]
            postxi[fill] = xi[uncensoredlist[0]]
            fill += 1
        print('count',count)
        count += 1
    
    return (postalpha, postbeta, postsigma, postcensoreta, postcensorxi, posteta, 
    postxi,uncensoredlist, censorlist, measurexc, measurexu, measureyu,etai,xi)

def obsdata(xi,eta,sigma, censor):
    #Simulates measured data given 'actual' data and measurement error variance
    
    #Input:
    #xi: Vector of covariates
    #eta: Vector of responses
    #sigma: 2x2 covariance matrix for measurement errors
    #censor: 1 to censor all values below 0, and 0 to keep all values
    
    #Output:
    #X: Vector of measured covariates
    #Y: Vector of measured responses
    
    err = np.random.multivariate_normal([0,0],sigma,len(xi))
    X = xi + err[:,0]
    Y = eta + err[:,1]
        
    if censor == 1:
        Y[Y < 0] = 0
    
    return X,Y

def kellysim(n, alpha, beta, var):
    #Draws n samples from the distribution given in the Kelly paper for 
    #simulated data. Drawing samples from a Uniform and Normal random variable
    #also present but commented out
    
    #Inputs:
    #n: Number of data points
    #alpha: Intercept term
    #beta: Regression coefficient
    #var: Variance of error terms
    
    #Output:
    #xi: Simulated covariate terms
    #eta: Simulated response terms
    
    #Covariates
    support = np.linspace(-4.7,5.5,500)
    vals = np.exp(support)/(1+np.exp(2.75*support))
    vals /= sum(vals)
    xi = np.random.choice(support,size=n,p=vals)
    #xi = np.random.normal(0,1,n)
    #xi = np.random.uniform(-1,1,n)
    
    #Error terms
    epsilon = np.random.normal(0,np.sqrt(var),n)
    
    eta = alpha + beta*xi + epsilon
    
    return xi, eta    
    
def convg(array,n,m):
    #Asseses parallel chains of the Gibbs sampler for convergence using the 
    #Gelman-Rubin Diagnostic
    
    #Input:
    #array: n*m array containing m chains each of length n
    #n: Length of each chain
    #m: Number of chains
    
    groupmean = np.mean(array, axis=0)
    W = np.mean(np.var(array,axis=0))
    B = n*np.mean(np.var(groupmean,axis=0))
    
    varplus = (n-1)*W/n + B/n
    Rhat = np.sqrt(varplus/W)
    
    return Rhat
 
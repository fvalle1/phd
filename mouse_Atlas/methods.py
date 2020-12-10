import numpy as np
import logging
logger = logging.getLogger()
sh = logging.Handler()
sh.setFormatter(logging.Formatter(fmt='%(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S'))
logger.addHandler(sh)


class method():
    def __init__(self, name=" ", color="gray", M=None, f=None):
        self.table = []
        self.f = f
        self.h = []
        self.M = M
        self.O = None
        self.__name__ = name
        self.color_ = color
        self.name_ = name
        
        self.hmean = None
        self.hvar = None
        self.cnt = None
        
    def get_p(self):
        return self.f/self.f.sum()
    
    def get_h(self):
        return self.h
    
    def get_f(self):
        f = np.array(self.table).T.mean(1)
        return np.sort(f/f.sum())[::-1]
    
    def get_O(self):
        if self.O is None:
            self.O =list(map(lambda x: len(x[x>0])/len(x), np.array(self.table).T))
        return self.O
    
    @property
    def matrix(self):
        return np.array(self.table)

    def sample(self, m) -> None:
        c = np.random.multinomial(m, self.get_p())
        self.table.append(c)
        self.h.append((c>0).sum())
    
    def run(self):
        logger.info(f"running {self.name_}")
        for m in self.M:
            self.sample(m)
    
    def __repr__(self):
        return self.__name__
    
class mazzolini(method):
    def __init__(self, *args, **kwargs):
        super().__init__("mazzolini", "orange", *args, **kwargs)
        
class mazzolini_pc(method):
    def __init__(self, pc=0.3, *args, **kwargs):
        super().__init__("mazzolini*P_c", "blue", *args, **kwargs)
        self.mu = pc
    
    def sample(self, m) -> None:
        c = np.random.multinomial(m, self.get_pvals())
        c[np.random.choice(len(c), round(0.3*len(c)))] = 0
        self.table.append(c)
        self.h.append((c>0).sum())
    
class mazzolini_broad(method):
    def __init__(self, M_tilde = 100000, *args, **kwargs):
        super().__init__("mazzolini_broad", "green", *args, **kwargs)
        self.p = None
        self.M_tilde = M_tilde
    
    def get_p(self):
        raise NotImplementedError("use get_pvals(m)")
        
    def get_pvals(self, m):
        if m > self.M_tilde:
            raise ValueError(f"{self.M} is a too low Mtilde use at least {m}")
        if self.p is None:
            self.p = np.array(
                [np.random.poisson(round(fi * self.M_tilde), 1)[0] for fi in super().get_p()])
            self.p = self.p/float(np.sum(self.p))
        return self.p
    
    def sample(self, m) -> None:
        try:
            c = np.random.multinomial(m, self.get_pvals(m))
            assert(c.sum()==m)
            self.table.append(c)
            self.h.append((c>0).sum())
        except:
            import sys
            print(sys.exc_info())
            print(np.sum(self.get_pvals()))
            print(np.isnan(self.get_pvals()).any())
            print((self.get_pvals(m)>=1).any())
            print((self.get_pvals(m)<0).any())
            
            
class mazzolini_nbinom(method):
    def __init__(self, M_tilde = 100000, *args, **kwargs):
        super().__init__("mazzolini_nbinom", "blue", *args, **kwargs)
        self.p = None
        self.M_tilde = M_tilde
    
    def get_p(self):
        raise NotImplementedError("use get_pvals(m)")
        
    def get_pvals(self, m):
        if m > self.M_tilde:
            raise ValueError(f"{self.M} is a too low Mtilde use at least {m}")
        if self.p is None:
            self.p = [np.random.negative_binomial(round(fi * self.M_tilde)/(round(fi * self.M_tilde)-1), 1/round(
                fi * self.M_tilde)) if round(fi * self.M_tilde) > 1 else 0 for fi in super().get_p()]
            self.p = self.p/np.sum(self.p)
        return self.p
    
    def sample(self, m) -> None:
        try:
            c = np.random.multinomial(m, self.get_pvals(m))
            self.table.append(c)
            self.h.append((c>0).sum())
        except:
            import sys
            print(sys.exc_info())
            
            
class mazzolini_gaus(method):
    def __init__(self, *args, **kwargs):
        super().__init__("mazzolini_gaus", "blue", *args, **kwargs)
        self.p = None
    
    def get_p(self):
        raise NotImplementedError("use get_pvals(m)")
        
    def get_pvals(self, m):
        if self.p is None:
            self.p = [np.clip(np.random.normal(fi,fi), 0, np.inf) for fi in super().get_p()]
            self.p = self.p/np.sum(self.p)
        return self.p
    
    def sample(self, m) -> None:
        try:
            c = np.random.multinomial(m, self.get_pvals(m))
            self.table.append(c)
            self.h.append((c>0).sum())
        except:
            import sys
            print(sys.exc_info())


class mazzolini_timesM(method):
    def __init__(self, multiplier = 10, *args, **kwargs):
        super().__init__(f"mazzolini_{multiplier}M", "blue", *args, **kwargs)
        self.p = None
        self.multiplier = multiplier
    
    def get_p(self):
        raise NotImplementedError("use get_pvals(m)")
        
    def get_pvals(self, m):
        if self.p is None:
            self.p = np.array([np.random.poisson(
                round(fi * self.multiplier * m), 1)[0] for fi in super().get_p()])
            self.p = self.p/float(np.sum(self.p))
        return self.p
    
    def sample(self, m) -> None:
        try:
            c = np.random.multinomial(m, self.get_pvals(m))
            assert(c.sum()==m)
            self.table.append(c)
            self.h.append((c>0).sum())
        except:
            import sys
            print(sys.exc_info())

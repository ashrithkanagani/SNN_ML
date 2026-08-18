import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

POLLUTANTS  = ["CO","NH3","NO2","OZONE","PM10","PM2.5","SO2"]
LABEL_NAMES = ["Good","Moderate","Poor","Severe"]
AQI_BREAKS  = {
    "PM2.5": [(30,0),(60,1),(90,2),(1e9,3)],
    "PM10":  [(50,0),(100,1),(250,2),(1e9,3)],
    "NO2":   [(40,0),(80,1),(180,2),(1e9,3)],
    "SO2":   [(40,0),(80,1),(380,2),(1e9,3)],
    "CO":    [(1, 0),(2, 1),(10, 2),(1e9,3)],
    "OZONE": [(50,0),(100,1),(168,2),(1e9,3)],
    "NH3":   [(200,0),(400,1),(800,2),(1e9,3)],
}

def load_data(path):
    df = pd.read_csv(path)
    print(f"[✓] Loaded  →  {df.shape[0]:,} rows  |  {df.shape[1]} columns")
    return df

def clean_data(df):
    df = df.drop_duplicates(); df.columns = df.columns.str.strip()
    df = df.dropna(subset=["pollutant_avg"])
    print(f"[✓] Cleaned →  {df.shape[0]:,} rows remaining")
    return df

def pollutant_score(val, pol):
    if pd.isna(val): return np.nan
    for thr, sc in AQI_BREAKS[pol]:
        if val <= thr: return float(sc)
    return 3.0

def engineer_features(df):
    
    pivot = df.pivot_table(
        index=["state","city","station","last_update","latitude","longitude"],
        columns="pollutant_id", values="pollutant_avg"
    ).reset_index(); pivot.columns.name = None

    for p in POLLUTANTS:
        if p not in pivot.columns: pivot[p] = np.nan
        pivot[f"sc_{p}"] = pivot[p].apply(lambda v: pollutant_score(v, p))

    sc_cols = [f"sc_{p}" for p in POLLUTANTS]
    pivot["mean_sc"]     = pivot[sc_cols].mean(axis=1)
    pivot["max_sc"]      = pivot[sc_cols].max(axis=1)
    pivot["n_avail"]     = pivot[POLLUTANTS].notna().sum(axis=1)
    pivot["ind_good"]    = (pivot["mean_sc"] < 0.5).astype(float)
    pivot["ind_poor"]    = (pivot["mean_sc"] >= 1.5).astype(float)
    pivot["ind_severe"]  = (pivot["mean_sc"] >= 2.5).astype(float)
    pivot["aqi_label"]   = pivot["mean_sc"].apply(
        lambda x: int(round(x)) if not pd.isna(x) else 1
    )

    feat_cols = (POLLUTANTS + sc_cols +
                 ["mean_sc","max_sc","n_avail",
                  "ind_good","ind_poor","ind_severe",
                  "latitude","longitude"])
    X = pivot[feat_cols].values.astype(float)
    y = pivot["aqi_label"].values.astype(int)

    dist = pd.Series(y).map({i:n for i,n in enumerate(LABEL_NAMES)}).value_counts()
    print(f"[✓] Pivoted  → {len(pivot)} snapshots  |  {len(feat_cols)} features")
    print(f"[✓] Class distribution:\n{dist.to_string()}\n")
    return X, y, feat_cols

def build_preprocessor(X_train, X_test):
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    X_train = sc.fit_transform(imp.fit_transform(X_train))
    X_test  = sc.transform(imp.transform(X_test))
    mn,mx = X_train.min(0), X_train.max(0)
    rng   = np.where(mx-mn < 1e-8, 1.0, mx-mn)
    X_train = np.clip((X_train-mn)/rng, 0, 1).astype(np.float32)
    X_test  = np.clip((X_test -mn)/rng, 0, 1).astype(np.float32)
    print(f"[✓] Preprocessed  X_train={X_train.shape}  X_test={X_test.shape}")
    return X_train, X_test

def rate_encode(X, T):
    return (np.random.rand(T, *X.shape) < X[None]).astype(np.float32)

class LIFLayer:
    """
    Leaky Integrate-and-Fire with soft reset and Adam optimiser.
    Surrogate gradient: piecewise-linear  max(0, 1 - |V - V_th|)
    """
    def __init__(self, n_in, n_out, tau_mem=20.0, threshold=0.5, lr=1e-3):
        self.W    = (np.random.randn(n_in,n_out)*np.sqrt(2.0/n_in)).astype(np.float32)
        self.b    = np.zeros(n_out, dtype=np.float32)
        self.alpha= np.float32(np.exp(-1.0/tau_mem))
        self.thr  = np.float32(threshold)
        self.lr   = lr
        self.mW=np.zeros_like(self.W); self.vW=np.zeros_like(self.W)
        self.mb=np.zeros_like(self.b); self.vb=np.zeros_like(self.b); self.t=0

    def forward(self, spk_in):
        T,B,_=spk_in.shape; n_out=self.W.shape[1]
        mem=np.zeros((B,n_out),dtype=np.float32)
        self._si=spk_in
        self._mh=np.empty((T,B,n_out),dtype=np.float32)
        self._sh=np.empty((T,B,n_out),dtype=np.float32)
        for t in range(T):
            mem=self.alpha*mem+spk_in[t]@self.W+self.b
            spk=(mem>=self.thr).astype(np.float32); mem-=self.thr*spk
            self._mh[t]=mem; self._sh[t]=spk
        return self._sh

    def backward(self, dout):
        T=dout.shape[0]; dW=np.zeros_like(self.W); db=np.zeros_like(self.b)
        din=np.zeros_like(self._si)
        for t in range(T):
            sg   =np.maximum(0.0, 1.0-np.abs(self._mh[t]-self.thr))
            delta=(dout[t]*sg).astype(np.float32)
            dW+=self._si[t].T@delta; db+=delta.sum(0); din[t]=delta@self.W.T
        self.t+=1; b1,b2,eps=0.9,0.999,1e-8
        for p,m,v,g_raw in[(self.W,self.mW,self.vW,dW/T),(self.b,self.mb,self.vb,db/T)]:
            g=np.clip(g_raw,-1.0,1.0).astype(np.float32)
            m[:]=b1*m+(1-b1)*g; v[:]=b2*v+(1-b2)*g*g
            mh=m/(1-b1**self.t); vh=v/(1-b2**self.t)
            p-=(self.lr*mh/(np.sqrt(vh)+eps)).astype(np.float32)
        return din

class SNN:
    def __init__(self, n_in, n_h1, n_h2, n_out, T=100, lr=1e-3, tau_mem=20.0):
        self.T  = T
        self.n_out = n_out
        self.l1 = LIFLayer(n_in, n_h1, tau_mem=tau_mem, threshold=0.5, lr=lr)
        self.l2 = LIFLayer(n_h1, n_h2, tau_mem=tau_mem, threshold=0.5, lr=lr)
        self.l3 = LIFLayer(n_h2, n_out, tau_mem=tau_mem, threshold=0.5, lr=lr)

    def _softmax(self, r):
        e=np.exp(r-r.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)

    def forward(self, spk_in):
        h1=self.l1.forward(spk_in); h2=self.l2.forward(h1); o=self.l3.forward(h2)
        return self._softmax(o.mean(0))

    def loss(self, probs, labels):
        return -np.log(np.clip(probs[np.arange(len(labels)),labels],1e-9,1.0)).mean()

    def backward(self, probs, labels):
        n=len(labels); g=probs.copy(); g[np.arange(n),labels]-=1.0; g/=n
        g_T=np.tile(g.astype(np.float32)[None],(self.T,1,1))/self.T
        gh2=self.l3.backward(g_T); gh1=self.l2.backward(gh2); self.l1.backward(gh1)

    def predict(self, X, mc_runs=20):
        """Average probability over mc_runs independent spike samples."""
        acc=np.zeros((len(X),self.n_out),dtype=np.float32)
        for _ in range(mc_runs):
            acc+=self.forward(rate_encode(X,self.T))
        return (acc/mc_runs).argmax(1)

def oversample_balanced(X, y):
    classes,counts=np.unique(y,return_counts=True); max_c=counts.max()
    Xs,ys=[],[]
    for cls,cnt in zip(classes,counts):
        idx=np.where(y==cls)[0]; rep=np.random.choice(idx,max_c,replace=(cnt<max_c))
        Xs.append(X[rep]); ys.append(np.full(max_c,cls,dtype=int))
    Xb=np.vstack(Xs); yb=np.concatenate(ys); p=np.random.permutation(len(yb))
    return Xb[p], yb[p]

def train_model(model, X_train, y_train, epochs=150, batch_size=64):
    X_b,y_b=oversample_balanced(X_train,y_train); n=len(X_b); history=[]
    for epoch in range(1, epochs+1):
        perm=np.random.permutation(n); X_s,y_s=X_b[perm],y_b[perm]
        total_loss,nb=0.0,0
        for s in range(0,n,batch_size):
            xb,yb=X_s[s:s+batch_size],y_s[s:s+batch_size]
            probs=model.forward(rate_encode(xb,model.T))
            total_loss+=model.loss(probs,yb); nb+=1; model.backward(probs,yb)
        avg=total_loss/nb; history.append(avg)
        if epoch%30==0 or epoch==1:
            acc=accuracy_score(y_train, model.predict(X_train, mc_runs=5))
            print(f"  Epoch {epoch:>4}/{epochs}  │  Loss: {avg:.4f}  │  Train Acc: {acc:.3f}")
    return history

def evaluate_model(model, X_test, y_test):
    preds=model.predict(X_test, mc_runs=20)
    acc=accuracy_score(y_test,preds)
    present=sorted(np.unique(np.concatenate([y_test,preds])))
    names=[LABEL_NAMES[i] for i in present]
    print("\n"+"═"*55)
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print("═"*55)
    print(classification_report(y_test,preds,labels=present,target_names=names,zero_division=0))
    cm=pd.DataFrame(confusion_matrix(y_test,preds,labels=present),index=names,columns=names)
    print("Confusion Matrix (rows=Actual, cols=Predicted):"); print(cm.to_string()); print("═"*55)
    return acc

def main():
    print("\n"+"═"*55)
    print("  Spiking Neural Network – India Air Quality (CPCB)")
    print("═"*55+"\n")

    df          = load_data("cpcbdata.csv")
    df          = clean_data(df)
    X,y,fcols   = engineer_features(df)
    print(f"  Features ({len(fcols)}): {fcols}\n")

    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    X_tr,X_te = build_preprocessor(X_tr,X_te)

    n_in=X_tr.shape[1]; n_h1,n_h2,n_out,T=256,128,4,100
    model=SNN(n_in=n_in,n_h1=n_h1,n_h2=n_h2,n_out=n_out,T=T,lr=1e-3,tau_mem=20.0)

    print(f"[✓] SNN: {n_in} → LIF({n_h1}) → LIF({n_h2}) → LIF({n_out})")
    print(f"    Neuron: LIF soft-reset  |  α={np.exp(-1/20.0):.4f}  |  V_th=0.5")
    print(f"    T={T} timesteps  |  Surrogate: piecewise-linear  |  Adam")
    print(f"    Inference: Monte Carlo spike averaging (20 runs)")
    print(f"    Features: raw pollutants + AQI scores + boundary indicators ({n_in} total)\n")
    print("── Training ──────────────────────────────────────────")

    train_model(model, X_tr, y_tr, epochs=150, batch_size=32)
    evaluate_model(model, X_te, y_te)
    pickle.dump(model,open("snn_model.pkl","wb"))

if __name__ == "__main__":
    main()
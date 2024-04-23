# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:23:15 2024

@author: user
"""
# =============================================================================
# Stress evolution plot
# =============================================================================
def time_plot(a,b,k):
    fig,(cax, ax) = plt.subplots(nrows=2,ncols=int(a), figsize=(int(a)*6,5.9),  
                                 gridspec_kw={"height_ratios":[0.5,10]})
    cbar_kws_ = {"shrink":.8,
               'extend':'both'} 
    for i in range(int(a)):
        sns.heatmap(b[i*3+47*k],vmin = 0,
                        cmap='RdYlBu_r' ,linecolor = 'k',linewidth=0,
                        ax = ax[i],cbar = False,
                        cbar_kws = dict(use_gridspec=True,location="top", extende = 'both',shrink = 0.3))
    #vmax = np.max(b[i*3+47*k])
        ax[i].set_xlabel(str(int(3*i+4))+'Δ', fontsize = 90)
    
    for i in range(a):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    # cax[0].axis("off")
    
    for i in range(a):
        fig.colorbar(ax[i].get_children()[0], cax=cax[i], orientation="horizontal", shrink=0.5)    
    
    # ax[0].set_xlabel("t$_\Delta$$_a$")
    # ax[1].set_xlabel("t$_\Delta$$_b$")
    # ax[2].set_xlabel("t$_\Delta$$_c$")
    # ax[3].set_xlabel("t$_\Delta$$_d$")
    
    
    for i in range(a):
        ax[i].spines['bottom'].set_visible(True)
        ax[i].spines['left'].set_visible(True)
        ax[i].spines['right'].set_visible(True)
        ax[i].spines['top'].set_visible(True)
        
        ax[i].spines['bottom'].set_linewidth(1.5)
        ax[i].spines['left'].set_linewidth(1.5)
        ax[i].spines['right'].set_linewidth(1.5)
        ax[i].spines['top'].set_linewidth(1.5)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)  
    

N=3
th_u=[0]
fi_u=[0]
lam_u=[pi/2]
th1=[pi,pi/2]
th2=[pi/2,pi]
th_v1=[pi,pi/2]
th_v2=[pi/2,pi]
fi_v1=[pi,pi/2]
fi_v2=[pi/2,pi]
lam_v1=[pi,pi/2]
lam_v2=[pi/2,pi]


def povmN(N,th_u,fi_u,lam_u,th1,th2,th_v1,th_v2,fi_v1,fi_v2,lam_v1,lam_v2):
    PovmN = QuantumCircuit(N,name='PovmN')

    PovmN.u(th_u[0],fi_u[0],lam_u[0],0)
    
    for i in range(1,N):
        R1 = QuantumCircuit(1,name='R1('+str(i)+')')
        R1.ry(th1[i-1],0)
        R_1=R1.to_gate().control(i)

        R2 = QuantumCircuit(1,name='R2('+str(i)+')')
        R2.ry(th2[i-1],0)
        R_2=R2.to_gate().control(i)
    
        PovmN.x(0)
        PovmN.append(R_1,range(i+1))
        PovmN.x(0)
        PovmN.append(R_2,range(i+1))
        
        V1 = QuantumCircuit(1,name='V1('+str(i)+')')
        V1.u(th_v1[i-1], fi_v1[i-1], lam_v1[i-1], 0)
        V_1=V1.to_gate().control(i)

        V2 = QuantumCircuit(1,name='V2('+str(i)+')')
        V2.u(th_v2[i-1], fi_v2[i-1], lam_v2[i-1], 0)
        V_2=V2.to_gate().control(i)
        
        PovmN.x(i)
        PovmN.append(V_1,list(range(1,i+1))+[0])
        PovmN.x(i)
        PovmN.append(V_2,list(range(1,i+1))+[0])
        
    return PovmN

POVMN=povmN(N,th_u,fi_u,lam_u,th1,th2,th_v1,th_v2,fi_v1,fi_v2,lam_v1,lam_v2)
POVMN.draw()

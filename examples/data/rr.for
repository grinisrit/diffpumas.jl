	subroutine rr(tss,fss,roct,rocs)
	real*4 roct,rocs,rock(1500),tteta,ffi
	open(unit=1,file='awork:rock.dat',
     *	form='formatted',status='old')
	read(1,105)rock
	close(unit=1)
 105	format(6E12.4)
	call oft(fss,tss,ksp)
	rocs=rock(ksp)*10
	tteta=tss
	ffi=fss
	call nmap(tteta,ffi,gsroc)
	roct=gsroc*2.71
	return
	end
*************************
	subroutine nmap(tteta,ffi,gsroc)
	include 'nm_c.inc'
	FFI=FFI+180.0
	tet=tteta
	fi=ffi
	if(ffi.ge.360.)fi=ffi-360.
	gsroc=-999999.
	iflag=1
		jt=tet/2.+1
		jf=fi/4.+1
		if(jt.ge.48.or.jf.ge.91)go to 300
	d1=dgsm(jt,jf)
		d2=dgsm(jt+1,jf)
			d3=dgsm(jt,jf+1)
				d4=dgsm(jt+1,jf+1)
	if(d1.gt.100000..or.d2.gt.100000..or.d3.gt.100000..or.d4.gt
     *	.100000.)iflag=-1
		if(d1.gt.100000.)d1=d1-100000.
		if(d2.gt.100000.)d2=d2-100000.
		if(d3.gt.100000.)d3=d3-100000.
		if(d4.gt.100000.)d4=d4-100000.
	t1=2.*(jt-1)
	f1=4.*(jf-1)
			r=(tet-t1)/2
			u=(fi-f1)/4.
	gsroc=(1.-r)*(1.-u)*d1+r*(1.-u)*d2+(1.-r)*u*d3+r*u*d4
c 100	continue
	if(iflag.lt.1)gsroc=gsroc+100000.
 300	xxx=0.0
	return
	end

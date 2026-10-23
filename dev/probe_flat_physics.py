#!/usr/bin/env python3
"""Read official C material moments through its public API; no transport or writes."""
import argparse,ctypes as c,csv,pathlib
p=argparse.ArgumentParser();p.add_argument('reference',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args()
lib=c.CDLL(str((a.reference/'libpumas.so.1').resolve()));libc=c.CDLL(None)
libc.fopen.argtypes=[c.c_char_p,c.c_char_p];libc.fopen.restype=c.c_void_p;libc.fclose.argtypes=[c.c_void_p]
physics=c.c_void_p();f=libc.fopen(str((a.reference/'materials-c.pumas').resolve()).encode(),b'rb');assert f
lib.pumas_physics_load.argtypes=[c.POINTER(c.c_void_p),c.c_void_p];assert lib.pumas_physics_load(c.byref(physics),f)==0;libc.fclose(f)
lib.pumas_physics_material_index.argtypes=[c.c_void_p,c.c_char_p,c.POINTER(c.c_int)]
with a.output.open('w') as stream:
 writer=csv.writer(stream);writer.writerow(['material','energy','property','value'])
 for name in ['StandardRock','Air']:
  material=c.c_int();assert lib.pumas_physics_material_index(physics,name.encode(),c.byref(material))==0
  for energy in [10.,100.,1000.,10000.,100000.]:
   for field,mode in [('stopping_power',0),('stopping_power',1),('range',0),('range',1),('transport_path',1),('proper_time',0),('energy_straggling',None),('cross_section',None)]:
    func=getattr(lib,'pumas_physics_property_'+field);value=c.c_double()
    if mode is None:
     func.argtypes=[c.c_void_p,c.c_int,c.c_double,c.POINTER(c.c_double)];rc=func(physics,material,energy,c.byref(value))
    else:
     func.argtypes=[c.c_void_p,c.c_int,c.c_int,c.c_double,c.POINTER(c.c_double)];rc=func(physics,mode,material,energy,c.byref(value))
    assert rc==0,(name,field,rc)
    writer.writerow([name,energy,field+('' if mode is None else '_'+str(mode)),value.value])
lib.pumas_physics_destroy.argtypes=[c.POINTER(c.c_void_p)];lib.pumas_physics_destroy(c.byref(physics))

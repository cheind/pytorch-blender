import bpy
import numpy as np


class Supershape:
    '''Three dimensional parametric supershape.
    
    See http://paulbourke.net/geometry/supershape/
    See http://wiki.theprovingground.org/blender-py-supershape
    '''
    def __init__(self, Unum=50, Vnum=50, smooth=True, name='supershape'):
        self.Unum = Unum
        self.Vnum = Vnum

        faces = self._faces()        
        verts = self._vertices()
        edges = []
        
#        print(verts.shape)

#        self.mesh = bpy.data.meshes.new(name)
#        self.mesh.from_pydata(verts.tolist(),edges,faces)
#        self.mesh.update(calc_edges=True)
#        self.obj = bpy.data.objects.new(name, self.mesh)
#        self.obj.location = (0,0,0)
#        self.obj.scale = (1,1,1)
#        bpy.context.collection.objects.link(self.obj)

#        #self.obj.modifiers.new("subd", type='SUBSURF')
#        #self.obj.modifiers['subd'].levels = 3

#        for p in self.mesh.polygons:
#            p.use_smooth = smooth

#        self.update()
        
    def update(self, m=0.01, a=1., b=1., n1=0.1, n2=0.01, n3=10.):
        verts = self._vertices(m,a,b,n1,n2,n3)
        for v,co in zip(self.mesh.vertices, verts):
            v.co = co
        
        bpy.context.view_layer.objects.active = self.obj
        #bpy.ops.object.mode_set(mode='EDIT')
        #bpy.ops.mesh.remove_doubles() 
        #bpy.ops.mesh.normals_make_consistent(inside=False)
        #bpy.ops.object.mode_set(mode='OBJECT')

    def _vertices(self, m=0.01, a=1., b=1., n1=0.1, n2=0.01, n3=10.):
        scale = 1.

        r = lambda alpha: (
            np.abs(np.cos(m*alpha/4.)/a)**n2 + 
            np.abs(np.sin(m*alpha/4.)/b)**n3
        )**(-1/n1)

        theta = np.linspace(-np.pi, np.pi, self.Unum)
        phi = np.linspace(-np.pi/2, np.pi/2, self.Vnum)
        
        u,v = np.meshgrid(phi, theta)
        print(u.shape, v.shape)
        uv = np.stack((u,v),-1)
        ruv = r(uv)

        r1 = r(theta)
        r2 = r(phi)

        x = r1*np.cos(theta)*r2*np.cos(phi)
        y = r1*np.sin(theta)*r2*np.cos(phi)
        z = r2*np.sin(phi)

        return np.stack((x,y,z), -1)
    
    def _vertices_old(self, m=0.01, a=1., b=1., n1=0.1, n2=0.01, n3=10.):
        scale = 1.
        
        Uinc = np.pi / (self.Unum/2)
        Vinc = (np.pi/2)/(self.Vnum/2)
        
        verts = []
        theta = -np.pi
        for i in range (0, self.Unum + 1):
            phi = -np.pi/2
            r1 = 1/(((abs(np.cos(m*theta/4)/a))**n2+(abs(np.sin(m*theta/4)/b))**n3)**n1)
            for j in range(0, self.Vnum + 1):
                r2 = 1/(((abs(np.cos(m*phi/4)/a))**n2+(abs(np.sin(m*phi/4)/b))**n3)**n1)
                x = scale * (r1 * np.cos(theta) * r2 * np.cos(phi))
                y = scale * (r1 * np.sin(theta) * r2 * np.cos(phi))
                z = scale * (r2 * np.sin(phi))
                vert = (x,y,z) 
                verts.append(vert)
                phi = phi + Vinc
            theta = theta + Uinc
        print(len(verts), (self.Unum + 1)*(self.Vnum+1))
        return verts

    def _faces(self):
        '''Returns the quads making up the supershape.'''
        faces = []
        count = 0
        maxidx = 0
        for i in range (0, (self.Vnum + 1) *(self.Unum)):
            if count < self.Vnum:
                A = i
                B = i+1
                C = (i+(self.Vnum+1))+1
                D = (i+(self.Vnum+1))    
                maxidx = max(A, maxidx)    
                maxidx = max(B, maxidx)
                maxidx = max(C, maxidx)
                maxidx = max(D, maxidx)                                                
                face = (A,B,C,D)
                faces.append(face)
                count = count + 1
            else:
                count = 0
        print(maxidx)
        return faces

def main():
    sshape = Supershape()

main()


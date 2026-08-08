# -*- coding: utf-8 -*-
import os
class fileSystem():
    def __init__(self, root='.', subdir='', caseStr='',**kwargs):
        self.record={}
        # default        
        self.register("history", os.path.join(root,'history/',subdir,'history'+caseStr), "txt" )
        self.register("historyfig", os.path.join(root,'history/',subdir,'history'+caseStr), "jpg" )
        self.register("track", os.path.join(root,'track/',subdir,'track'+caseStr), "csv" )
        self.register("figTrack", os.path.join(root,'track/',subdir,'fig'+caseStr), "jpg" )
        self.register("figTDOATrack", os.path.join(root,'tdoa/',subdir,'fig'+caseStr), "jpg" )
        self.register("log", os.path.join(root,'log/',subdir,'log'+caseStr), "txt" )
        self.register("params", os.path.join(root,'params/',subdir,'track'+caseStr), "json" )
        self.register("net", os.path.join(root,'net/',subdir,'net'+caseStr), "net" )
        self.register("error", os.path.join(root,'track/',subdir,'error'+caseStr), "pickle" )
        self.register("figDetection", os.path.join(root,'detections/',subdir,''), "jpg" )
        self.register("preSync", os.path.join(root,'preSync/',subdir,''), "jpg" )
        self.defaultFlag=list( self.record.keys())
        for flag, info in kwargs.items():
            assert isinstance(info, tuple) and len(info)==2
            file_prefix, scheme = info
            self.register(flag, file_prefix, scheme)

    def register(self, flag, file_prefix, scheme):
        if flag  in self.record and flag not in self.defaultFlag:
            print(f"Warning, file flag <{flag}> has been overwritten")
        self.prepare(file_prefix)
        self.record[flag] = (file_prefix, scheme)
        return self

    def prepare(self,prefix):
        father_path=os.path.dirname(prefix)
        if not os.path.isdir(father_path):
            os.makedirs(father_path)
        return prefix
    def __add(self, flag, name):
        add = name if name=="" else "_"+name
        file_prefix, scheme = self.record[flag]
        return file_prefix + add + "." + scheme

    def __getattr__(self,flag):
        if flag not in self.record:
            raise Exception(f"Unknow file flag <{flag}>")
        def fun(name=""):
            return self.__add(flag, name=name)
        setattr(self, flag, fun)
        return getattr(self,flag)

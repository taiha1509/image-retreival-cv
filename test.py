import os, glob

path = 'static/dataset'
dict_categories = ["art_1","art_antiques","art_cybr","art_dino","art_mural","bld_castle","bld_lighthse","bld_modern","bld_sculpt","eat_drinks","eat_feasts","fitness","obj_234000","obj_aviation","obj_balloon","obj_bob","obj_bonsai","obj_bus","obj_car","obj_cards","obj_decoys","obj_dish","obj_doll","obj_door","obj_eastregg","obj_flags","obj_mask","obj_mineral","obj_moleculr","obj_orbits","obj_ship","obj_steameng","obj_train","pet_cat","pet_dog","pl_flower","pl_foliage","pl_mashroom","sc_","sc_autumn","sc_cloud","sc_firewrk","sc_forests","sc_iceburg","sc_indoor","sc_mountain","sc_night","sc_rockform","sc_rural","sc_sunset","sc_waterfal","sc_waves","sp_ski","texture_1","texture_2","texture_3","texture_4","texture_5","texture_6","wl_buttrfly","wl_cat","wl_cougr","wl_deer","wl_eagle","wl_elephant","wl_fish","wl_fox","wl_goat","wl_horse","wl_lepoad","wl_lion","wl_lizard","wl_nests","wl_owls","wl_porp","wl_primates","wl_roho","wl_tiger","wl_wolf","woman"]
count_each_class = {}
i=0
for label in dict_categories:
    c = 0
    for filename in os.listdir(path):
        if label in filename:
            c+=1
    count_each_class[i] = c
    i+=1
print(count_each_class)

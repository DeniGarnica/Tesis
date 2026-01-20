Pipeline

0. tener los frames a ser trackeados en la carpeta `data` con el nombre deseado de esta base de datos
        Ej: `data/axolot`


La siguiente seccion de pasos se puede ejjecutar en automatico corriendo el archivo 
                `VGGT-Cotracker/main_part1.py` base_de_datos
Donde base de datos es el nombre de la carpeta de frames. Es imoprtante tener un env conda llamado `Samv2` el cual tiene instalado todo lo necesario para correr Samv2.

        1. Correr el script de python `VGGT-Cotracker/1-vggt_outputs_run.py` el cual creara una carpeta de outputs
        2. Correr el script de python `VGGT-Cotracker/2-Cotracker_run.py` el cual creara los puntos a seguir 
        3. Correr el script de python `VGGT-Cotracker/3-sam_masks_run.py` con el env activado para `Samv2` el cual creara las mascaras de los objetos a seguir. Estas fueron inicializadas con los puntos del Cotracker, por lo que no sera posible realizar esto sin el paso anterior.

        4. Se seleccionaran las mascras a usar manualmente, las cuales estan en la carpeta `masks_all`. Sus nombres se escribiran en el argumento `masks = []` del archivo `VGGT-Cotracker/4-use_mask.py`. (Unico paso manual, se espera hacer automatico eventualmente) y se correra este script 

        5. Correr el script de python `VGGT-Cotracker/TrackTo3D.py` el cual nos devolvera el archivo `outputs/axolot/cotracked_points_3d_world.csv`
        6. Correr el script de python `VGGT-Cotracker/to3D.py` el cual nos dara los resultados finales de trackeo

Nota: falta crear un requirements a instalar para el resto de los archivos


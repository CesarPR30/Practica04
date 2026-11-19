import html, os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from PIL import Image
D=os.path.dirname(os.path.abspath(__file__)); R=os.path.dirname(D)
G="Grupo: Ricardo Acuña, César Pajuelo, Angello Soldi, Josué Velo, Paolo Medrano"
S=[("Pipeline de ML de extremo a extremo para reclamaciones de seguros",None,["Isolation Forest, MLflow y BentoML",G]),
("El problema: un flujo de revisión completamente manual","figure pracitca 4.png",["Cada reclamación pasa por el comité, aunque la mayoría es rutinaria.","El modelo actúa como filtro previo: aprueba lo normal y deriva lo atípico al equipo de investigación.","Resultado: menos carga del comité y foco en los casos de riesgo."]),
("Datos: la anomalía vive en tres variables","capturas/01_terminal_generacion_datos.png",["Lo anómalo combina montos altos, muchos servicios y reclamos muy seguidos.","Edad y proveedor se distribuyen igual en ambas clases: no aportan señal.","Incluir variables sin señal diluye el aislamiento del bosque."]),
("Experimentación trazable en MLflow","capturas/04_mlflow_comparacion_runs.png",["Seis runs con parámetros, métricas, matriz de confusión y modelo registrados.","Las líneas que pasan por el núcleo de variables terminan en el F1 más alto.","El umbral (contamination) equilibra fraude no detectado frente a carga del comité."]),
("Mejora: la selección de variables pesa más que los hiperparámetros","capturas/11_comparacion_f1_kfold.png",["Quitar edad y proveedor sube el F1 en validación cruzada de 0.887 a 0.940.","La variabilidad entre pliegues baja de 0.135 a 0.081: el modelo es más estable.","Las variables sin señal consumen cortes del bosque que no aíslan nada."]),
("¿Sobreajuste? La validación cruzada dice que no","capturas/06_matriz_confusion_mejor_run.png",["El modelo se elige con validación cruzada 5x3, sin mirar el conjunto de prueba.","F1 de 1.0 en entrenamiento frente a 0.94 en validación: brecha pequeña.","El 1.0 en prueba es una partición favorable de 45 casos; la estimación realista es 0.94.","El AUC de 1.0 se debe a datos sintéticos sin solapamiento; con datos reales será menor."]),
("Registro con alias champion","capturas/07_mlflow_model_registry.png",["El servicio apunta al alias champion, no a una versión fija.","Promover un modelo nuevo es reasignar el alias sin tocar el servicio."]),
("Servicio BentoML: carga, análisis y decisión","capturas/10_terminal_prediccion_csv.png",["Carga masiva por CSV o JSON, validada antes de llegar al modelo.","Se derivan a revisión justo los casos con monto alto, muchos servicios y pocos días.","El puntaje ordena la cola de revisión: la reclamación más atípica se atiende primero.","Los casos normales quedan más lejos del umbral que con el modelo anterior: decisiones más estables."]),
("Conclusiones",None,["Pipeline completo y reproducible: datos, experimentos, registro y servicio.","La selección de variables pesó más que los hiperparámetros: F1 de 0.887 a 0.940.","El umbral es una decisión de negocio: recall frente a carga del comité.","Siguiente paso: datos reales con solapamiento, monitoreo de drift y explicabilidad."])]
p=Presentation(); p.slide_width=Inches(13.33); p.slide_height=Inches(7.5); N=RGBColor(0x1F,0x2A,0x44)
for t,img,pts in S:
    s=p.slides.add_slide(p.slide_layouts[6])
    b=s.shapes.add_shape(1,0,0,p.slide_width,Inches(0.9)); b.fill.solid(); b.fill.fore_color.rgb=N; b.line.fill.background()
    tf=s.shapes.add_textbox(Inches(0.4),Inches(0.12),Inches(12.5),Inches(0.7)).text_frame; tf.text=t; r=tf.paragraphs[0].runs[0]; r.font.size=Pt(26); r.font.bold=True; r.font.color.rgb=RGBColor(255,255,255)
    w=5.3 if img else 12.3
    tf=s.shapes.add_textbox(Inches(0.4),Inches(1.3),Inches(w),Inches(5.6)).text_frame; tf.word_wrap=True
    for i,x in enumerate(pts):
        para=tf.paragraphs[0] if i==0 else tf.add_paragraph(); para.text="• "+x; para.font.size=Pt(18); para.space_after=Pt(12)
    if img:
        iw,ih=Image.open(os.path.join(R,img)).size; sc=min(7.0/iw,5.6/ih)
        s.shapes.add_picture(os.path.join(R,img),Inches(5.95),Inches(1.3),Inches(iw*sc),Inches(ih*sc))
p.save(os.path.join(D,"presentacion_practica4.pptx"))
e=html.escape
sec="".join(f'<section><h2>{e(t)}</h2><div class="c"><ul>{"".join(f"<li>{e(x)}</li>" for x in pts)}</ul>{f"<img src=\"../{img.replace(" ","%20")}\">" if img else ""}</div></section>' for t,img,pts in S)
open(os.path.join(D,"presentacion_practica4.html"),"w").write(f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Práctica 4 Pipeline ML</title><style>body{{margin:0;font-family:Calibri,Arial,sans-serif;background:#eef1f6}}section{{background:#fff;max-width:1100px;margin:24px auto;border-radius:10px;overflow:hidden;box-shadow:0 2px 10px #0002}}h2{{margin:0;background:#1F2A44;color:#fff;padding:16px 24px}}.c{{display:flex;gap:20px;padding:20px 24px;flex-wrap:wrap}}.c ul{{flex:1 1 320px;font-size:18px;line-height:1.5}}.c img{{flex:1 1 420px;max-width:100%;object-fit:contain;max-height:480px}}</style></head><body>{sec}</body></html>''')

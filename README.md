# ampel-renderer

![Screen Shot 2022-03-09 at 23 25 04](https://user-images.githubusercontent.com/38134006/157552898-5c21b137-e1cf-4140-8ad6-9c41ce231fc6.png)

## Download
Hier kann das Programm kompiliert heruntergeladen werden: <LINK

## Grundlagen
### Renderer
Der Renderer wurde ohne jegliche vorgefertigte Softwarebibliotheken programmiert - einzig die Abstraktionsebene `wgpu` wird verwendet, um plattformübergreifende Kompatibilität zu gewährleisten. `wgpu` bildet einen Draht zu den nativen GPU-APIs der jeweiligen Plattform, so wird auf Linux bspw. OpenGL bzw. Vulkan als Implementierung verwendet und auf Windows DirectX.

Beim Umgang mit Grafik-Hardware sind sog. Buffers zentral. Sie sind kleine Abschnitte im Arbeitsspeicher von Grafikkarten. Ein Programm kann solche Buffers reservieren und Daten an sie senden. Dabei werden die Buffers bei jedem Renderdurchgang (d.h. bei jedem Frame) überschrieben und müssen vom Programm neu versandt werden.

Auf der GPU läuft ebenfalls Code - in Form von sog. Shadern. Ein Shader ist ein sehr kleines, hoch parallelisierbares Programm, das das eigentliche 3D-Modell auf den Bildschirm zeichnet. Dabei sind vor allem zwei Arten von Shadern im Einsatz: Vertex Shader und Fragment Shader (es wird also ein Shader pro Fragment ausgeführt). Erstere sind für die Positionierung der Vertices verantwortlich (z.B. können hier Vertices gedreht, skaliert, verzerrt usw. werden), letztere für das Berechnen der Farbe eines *Fragments* (meist ein Pixel groß).
In den Shadern können dann die Daten, die in den Buffers gespeichert werden, verarbeitet werden. So lässt sich beispielsweise die Farbe eines Objekts jeden Frame ändern, indem in jedem Renderdurchgang der Buffer auf der Grafikkarte verändert und anhand der übermittelten Daten die Farbe des jeweiligen Fragments verändert wird.

Ein Buffer hat eine Adresse, einen Pointer, die sog. BindGroup. BindGroups sind von Nöten, um vom CPU-seitigen Code Buffers an die GPU zu senden. Meherere BindGroups können Teil einer *RenderPipeline* sein - diese fasst alle Eigenschaften und Übertragungsmethoden eines Shaders zusammen und lässt sich spontan austauschen, bspw. um den Shader zu wechseln. Beim Übertragen von Daten an die GPU ist es zudem wichtig, bestimmte Memory Alignment-Regeln einzuhalten. Nur, wenn ein bestimmter Abstand zwischen bestimmten Datentypen in einem Uniform (Übertragungsmittel an den Shader) gewährleistet ist, kann die GPU ihren Arbeitsspeicherhaushalt optimisieren.

### 3D-Dateiformate
Das von meinem Renderer verwendete Dateiformat ist `Wavefront OBJ`. Dabei handelt es sich um `.obj`-Dateien, die 
#### Model
Das mit der OBJ-Datei beschriebene 3D-Modell.

#### Mesh
Eine OBJ-Datei bzw. Model kann mehrere sog. Meshes enthalten. Diese bestehen wiederum aus mehreren *Vertices* (Punkten im 3D-Raum), die miteinander zum Mesh verbunden werden. Ein Mesh kann man sich als "Oberfläche" zwischen mehreren Vertices vorstellen.

#### Material
Jedes Mesh besitzt ein Material. Das Material gibt die Beschaffenheit eines Meshs an - welche Farbe hat es, wie sehr reflektiert es Licht, etc.

In OBJ-Dateien können sich mehrere Meshes ein Material teilen, aber jedes Mesh darf nur ein Material besitzen.


### Berechnung von Licht
*zum Code: [Berechnung des Lichts in shader.wgsl](/renderer/src/shader.wgsl)*


Man bräuchte einen extrem leistungsfähigen Rechner, um Licht nach den physikalischen Gesetzen akkurat darzustellen - geschweige denn in Echtzeit. Deshalb gibt es verschiedene Annäherungen an die Wirklichkeit, die sehr viel weniger leistungshungrig sind. So wird in diesem Renderer bspw. das Blinn-Phong-Modell verwendet, um den Einfluss von Licht auf einem Objekt zu berechnen.

Dabei ist nicht die Lichtquelle für die Farbveränderung auf einem benachbarten Objekt verantwortlich, sondern dieses Objekt selbst: In jedem Fragment des Objekts wird mithilfe einer mathematischen Formel unter Einbeziehung der Entfernung, des Winkels und der Eigenschaften der Lichtquelle die Farbe berechnet, die das Fragment haben soll.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Phong_components_version_4.png/655px-Phong_components_version_4.png)

*Die Zusammensetzung der Reflektion nach Blinn-Phong*

Dabei besteht die Reflektion aus drei Teilen: Ambient, Diffuse und Specular. Ambient ist eine Annäherung an die Tatsache, dass Lichtstrahlen unendlich oft an Objekten in der Realität abprallen - selbst im Schatten ist es nie vollständig schwarz. Deshalb gibt man einem Objekt eine dunkle Grundfarbe. Diffuse bezeichnet den Teil des Lichts, der vom Objekt reflektiert wird. Dabei kommt es bei der Stärke auf den Winkel zur Lichtquelle an. Specular bezeichnet die spiegelnde Lichtkomponente. Diese wird in Abhängigkeit zur Kamera, also zum Betrachter, berechnet. 

Die Lampen selbst werden übrigens mit einem anderen Shader in [light.wgsl](/renderer/src/light.wgsl) gerendert, da diese nicht von anderen Lampen bzw. Schatten beeinflusst werden sollen, schließlich leuchten sie.

## Konfiguration
Die Szene beim Start des Programms deserialisiert aus der Datei [world.toml](/renderer/res/world.toml). Hier werden Informationen über alle Modelle in der Szene gespeichert, so z.B. der Name eines Modells, eine einzigartige ID, die Position in der Szene, die Rotation und der Pfad zur OBJ-Datei.

Beim Klick auf *Save world* wird die sich im Arbeitsspeicher befindliche Welt serialisiert und in die Datei geschrieben.

## Bedienung
Die Kamera muss vom Benutzer so eingestellt werden, dass die Szene sichtbar ist.
### Kamera
Die Kamera kann mit <kbd>A</kbd> nach links und mit <kbd>D</kbd> nach rechts um den Ursprung gedreht werden.

Mit <kbd>W</kbd> kann hereingezoomt und mit <kbd>S</kbd> herausgezoomt werden.

### Benutzeroberfläche
Wenn das Programm geöffnet wird, fallen zwei Fenster auf: Eines mit dem Titel "Settings", das andere mit dem Titel "Scene".

#### Settings

* **Clear color**: Mit einem Klick auf den Button erscheint eine Farbauswahl. Mit dieser Einstellung kann die Clear Color (Hintergrundfarbe, die sichtbar ist, wenn nichts an einer Stelle gerendert wird) eingestellt werden.
* **Only show emissive materials**: Falls aktiv werden ausschließlich lichtemittierende Materials angezeigt.
* **Reset Camera**: Setzt die Position der Kamera zurück zum Ursprung.
* **Reload world**: Lädt die Szene neu von der Konfigurationsdatei.
* **Save world**: Speichert die Szene in die Konfigurationsdatei.
* **Green/Yellow/Red duration**: Mit diesen Schiebereglern lassen sich die Dauer der einzelnen Ampelphasen einstellen. Einheit: Sekunden
* **Start / Stop**: Startet bzw. stoppt die Ampelsteuerung.

#### Scene
Hier lassen sich alle Eigenschaften der Objekte in der Szene einstellen. Auf der äußersten Ebene des Baumdiagramms werden die einzelnen Models angezeigt. Durch einen Klick auf den Pfeil neben einem Model lassen sich die Einstellungen des Models öffnen. Im Reiter "Model" lassen sich Position und Rotation des gesamten Models einstellen. 
Die nachfolgenden Reiter bieten Einstellmöglichkeiten für die im Model enthaltenen Meshes. Lichtemittierende Meshes bieten zudem hier die Möglichkeit, das ihnen zugeordnete Material anzupassen.

## Ampelsteuerung
*zum Code: [Ampel -Klasse](/renderer/src/ampel.rs) | [Steuerung](/renderer/src/main.rs)*

Durch einen Klick auf **Start / Stop** lässt sich die Ampelsteuerung starten. Durch die Reflektion der Ampellichter auf der Straße ist die derzeitige Ampelphase aus jedem Winkel erkennbar, aber mit <kbd>A</kbd> bzw. <kbd>D</kbd> kann die Kreuzung unter allen Winkeln betrachtet werden.


# ampel-renderer

![Screen Shot 2022-03-09 at 23 25 04](https://user-images.githubusercontent.com/38134006/157552898-5c21b137-e1cf-4140-8ad6-9c41ce231fc6.png)

## Download
Hier kann das Programm kompiliert heruntergeladen werden: [Download (Windows)](https://github.com/realmayus/wgpu_renderer/raw/master/renderer_neu.zip)

Es ist wichtig, dass das Programm mit derselben Ordnerstruktur wie in der Zip-Datei entpackt wird. D.h. im selben Verzeichnis wie die EXE-Datei muss der Ordner `res` liegen. Das Programm benutzt relative Pfade beim Laden der Ressourcen und nur so kann gewährleistet werden, dass die benötigten Dateien gefunden werden.

Falls sich das Programm direkt wieder schließt, kann es sein, dass ein Fehler aufgetreten ist. Diesen kann man sich anzeigen lassen, indem man das Programm durch die Kommandozeile öffnet. Dazu einfach mit dem Terminal in den Ordner der EXE-Datei springen und die EXE übers Terminal ausführen. Dann sollte in der Konsole ein Fehler angezeigt werden.

## Grundlagen
### Wichtig
Ich habe hier versucht, die Grundideen in meinem Programm etwas zu erläutern. Da es aber sehr komplex ist, ist es schwer, hier alles zu berücksichtigen. Vor allem kann ich Ihnen in der nächsten Stunde auch den Code näher erklären (Rust als Programmiersprache hat einige interessante Eigenschaften). Falls jedoch Fragen bestehen, können Sie mir gerne eine Email schreiben.

### Rust
Der Renderer wurde in der Programmiersprache Rust entwickelt. Rust ist so nah an der Hardware wie z. B. C++, ist aber deutlich ergonomischer in der Handhabung und gewährleistet Memory Safety, d.h. dass es schlichtweg unmöglich ist, einen bereits vom Arbeitsspeicher gelöschten Wert zu lesen. Dadurch werden viele Fehler verhindert und können überhaupt nicht auftreten.
### Renderer
Der Renderer wurde ohne jegliche vorgefertigte Softwarebibliotheken programmiert - einzig die Abstraktionsebene `wgpu` wird verwendet, um plattformübergreifende Kompatibilität zu gewährleisten. `wgpu` bildet einen Draht zu den nativen GPU-APIs der jeweiligen Plattform, so wird auf Linux bspw. OpenGL bzw. Vulkan als Implementierung verwendet und auf Windows DirectX.

Beim Umgang mit Grafik-Hardware sind sog. Buffers zentral. Sie sind kleine Abschnitte im Arbeitsspeicher von Grafikkarten. Ein Programm kann solche Buffers reservieren und Daten an sie senden. Dabei werden die Buffers bei jedem Renderdurchgang (d.h. bei jedem Frame) überschrieben und müssen vom Programm neu versandt werden.

Auf der GPU läuft ebenfalls Code - in Form von sog. Shadern. Ein Shader ist ein sehr kleines, hoch parallelisierbares Programm, das das eigentliche 3D-Modell auf den Bildschirm zeichnet. Dabei sind vor allem zwei Arten von Shadern im Einsatz: Vertex Shader und Fragment Shader (es wird also ein Shader pro Fragment ausgeführt). Erstere sind für die Positionierung der Vertices verantwortlich (z.B. können hier Vertices gedreht, skaliert, verzerrt usw. werden), letztere für das Berechnen der Farbe eines *Fragments* (min. ein Pixel groß).
In den Shadern können dann die Daten, die in den Buffers gespeichert werden, verarbeitet werden. So lässt sich beispielsweise die Farbe eines Objekts jeden Frame ändern, indem in jedem Renderdurchgang der Buffer auf der Grafikkarte verändert und anhand der übermittelten Daten die Farbe des jeweiligen Fragments verändert wird.

Ein Buffer hat eine Adresse, einen Pointer, die sog. BindGroup. BindGroups sind von Nöten, um vom CPU-seitigen Code Buffers an die GPU zu senden. Meherere BindGroups können Teil einer *RenderPipeline* sein - diese fasst alle Eigenschaften und Übertragungsmethoden eines Shaders zusammen und lässt sich spontan austauschen, bspw. um den Shader zu wechseln. Beim Übertragen von Daten an die GPU ist es zudem wichtig, bestimmte Memory Alignment-Regeln einzuhalten. Nur, wenn ein bestimmter Abstand zwischen bestimmten Datentypen in einem Uniform (Übertragungsmittel an den Shader) gewährleistet ist, kann die GPU ihren Arbeitsspeicherhaushalt optimisieren.

```rs
let uniform = LightUniform {   // (1)
    ambient: mat.ambient,
    constant: 1.0,
    diffuse: mat.diffuse,
    linear: 0.09,
    specular: mat.specular,
    quadratic: 0.032,
};

uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {  // (2)
    label: Some("LightUniform Buffer"),
    contents: bytemuck::cast_slice(&[uniform]),
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
});

bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {   // (3)
    layout: &light_layout,
    entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform_buffer.as_entire_binding(),
    }],
    label: None,
});
```
*Beispiel: Anordnen der Daten, die an die GPU gesendet werden (1), erstellen des Buffers auf der GPU (reservieren des Speichers) (2), erstellen der BindGroup, damit eine "Route" zwischen GPU-seitigem Buffer und CPU-seitigem Code entsteht (3)*

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

```
    let lightDir: vec3<f32> = normalize(((light.position * rot_mat) + light.model_offset + light.worldpos) - (in.world_position));
    let diff: f32 = max(dot(in.world_normal, lightDir), 0.0);
    var diffuse: vec3<f32> = light.diffuse * diff * obj_color;
```
*Beispiel: Berechnung der Diffuse-Farbe eines Fragments* 

Die Lampen selbst werden übrigens mit einem anderen Shader in [light.wgsl](/renderer/src/light.wgsl) gerendert, da diese von allen Winkeln aus die volle Farbe haben sollen - schließlich leuchten sie.

## Dateistruktur
Der Code liegt in [renderer/src](renderer/src). Die Ressourcen, die vom Programm geladen werden, liegen in [renderer/res](renderer/res) (Model-Dateien, Materials, Texturen usw.)
### Konfigurationsdatei

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
*zum Code: [Ampel -Klasse](/renderer/src/ampel.rs) | [Steuerung](https://github.com/realmayus/wgpu_renderer/blob/b1d530be185552076a3f79b3ee119f65661d316c/renderer/src/main.rs#L654)*

Durch einen Klick auf **Start / Stop** lässt sich die Ampelsteuerung starten. Durch die Reflektion der Ampellichter auf der Straße ist die derzeitige Ampelphase aus jedem Winkel erkennbar, aber mit <kbd>A</kbd> bzw. <kbd>D</kbd> kann die Kreuzung unter allen Winkeln betrachtet werden.

Die Ampelsteuerung liegt in der Funktion `update()`, die bei jedem Renderdurchgang, also bei jedem Frame, ausgeführt wird. Als Klassenattribut wird ein Ampel-Index (`ampel_index`) sowie ein Zeitpunkt gespeichert, bei dem in die nächste Phase übergegangen werden soll (`next_cycle`).

```rs
        if SystemTime::now().duration_since(UNIX_EPOCH).unwrap() > self.next_cycle {
```
Bei jedem Frame wird hier gecheckt, ob sich die derzeitige Systemzeit nach dem definierten `next_cycle` befindet. Falls dies der Fall ist, also ein Phasenübergang überfällig ist, werden anhand des `ampel_index` die Ampeln umgeschaltet:

```rs
                    State::set_ampel_status(
                        State::find_model(
                            self.models.as_mut_slice(),
                            "a0c49f2b-5e48-48ee-85a2-86a88900617f".to_string(),
                        ),
                        AmpelStatus::RED,
                    );
```
Das Ampel-Model wird anhand einer UUID (universally unique identifier) im Programm identifiziert und dessen Materials werden so verändert, dass es aussieht, als würden z.B. die gelbe und die rote Lampe ausgeschaltet, und die Grüne eingeschaltet sein:
```rs
        match status {
            AmpelStatus::RED => {
                mat_red.as_mut().map(|mut s| {
                    s.uniform.quadratic = 0.032;
                    s
                });
                mat_red.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.8, 0.0, 0.011073];
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_yellow.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.033, 0.031, 0.004];
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.quadratic = 100.0;
                    s
                });
                mat_green.as_mut().map(|mut s| {
                    s.uniform.diffuse = [0.002, 0.027, 0.002];
                    s
                });
            }
```

Dabei sind immer zwei gegenüberliegende Ampeln gekoppelt und sind in der gleichen Phase.
Die Phasenlängen sind über die Slider im GUI einstellbar.
Damit ein Start/Stop-Button implementiert werden konnte, wurde zudem ein Ampelindex `-1`eingeführt. Wenn dieser Index aktiv ist, werden alle Ampeln auf Rot gestellt, aber im Gegensatz zu anderen Indizes wird kein nachfolgender Index bestimmt. Beim Klick auf Start/Stop wird Index `0` aktiv, der ebenfalls alle Ampeln auf Rot setzt, jedoch einen Wert für `next_cycle` definiert und auch den nächsten Index auf `1` setzt.
Wenn die Schaltung gestoppt wird, wird wieder der Ampelindex `-1` gesetzt.

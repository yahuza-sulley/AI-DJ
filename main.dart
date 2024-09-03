import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:music21/music21.dart';

void main() {
  runApp(MusicComposerApp());
}

class MusicComposerApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Music Composer',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: MusicComposerPage(),
    );
  }
}

class MusicComposerPage extends StatefulWidget {
  @override
  _MusicComposerPageState createState() => _MusicComposerPageState();
}

class _MusicComposerPageState extends State<MusicComposerPage> {
  final String modelUrl =
      'https://example.com/best_model.tflite'; // Replace with your model URL
  String generatedMusicFileUrl;
  TextEditingController seedNoteController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('AI Music Composer')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: seedNoteController,
              decoration: InputDecoration(labelText: 'Enter seed note'),
            ),
            ElevatedButton(
              onPressed: generateMusic,
              child: Text('Generate Music'),
            ),
            if (generatedMusicFileUrl != null)
              ElevatedButton(
                onPressed: playGeneratedMusic,
                child: Text('Play Generated Music'),
              ),
          ],
        ),
      ),
    );
  }

  void generateMusic() async {
    String seedNote = seedNoteController.text;

    // Send the seed note to your AI model API (Replace with your API URL)
    // For example, you can use http package to make an HTTP POST request
    String apiUrl = 'https://example.com/generate_music';
    var response = await http.post(apiUrl, body: {'seed_note': seedNote});

    if (response.statusCode == 200) {
      // Parse the response to get the URL of the generated music MIDI file
      setState(() {
        generatedMusicFileUrl = response.body;
      });
    } else {
      print('Failed to generate music. Status code: ${response.statusCode}');
    }
  }

  void playGeneratedMusic() async {
    // Download the generated music MIDI file from the URL
    var bytes = await http.readBytes(Uri.parse(generatedMusicFileUrl));

    // Save the MIDI file to the device's temporary directory
    Directory tempDir = await getTemporaryDirectory();
    File tempFile = File('${tempDir.path}/generated_music.mid');
    await tempFile.writeAsBytes(bytes);

    // Load and play the MIDI file using music21
    var midiData = File('${tempDir.path}/generated_music.mid').readAsBytesSync();
    var midiStream = Stream.fromIterable(midiData);
    var midiStreamIterator = midiStream.iterator;
    midi.StreamScore(scoreFormat: midi.MidiFile(midiStreamIterator)).play();
  }
}

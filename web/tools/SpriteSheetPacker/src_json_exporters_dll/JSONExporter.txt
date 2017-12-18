using System.Collections.Generic;
using System.Drawing;
using System.IO;


namespace SpriteSheetPacker.JsonExporters
{
    //Creates a JSON Hash Array for use in compatible game engines 
    public class JSONExporter : sspack.IMapExporter
    {
        public string MapExtension
        {
            get { return "json"; }
        }

        public void Save(string filename, Dictionary<string, Rectangle> map)
        {
            using(StreamWriter writer = new StreamWriter(filename))
            {
                writer.WriteLine("{\"frames\":{");
                writer.WriteLine("");

                int totalDictionaryCount = 0; 

                foreach(var entry in map)
                {
                    Rectangle rect = entry.Value;
                    writer.WriteLine("\""+Path.GetFileName(entry.Key)+"\":");
                    writer.WriteLine("{");
                    writer.WriteLine(string.Format("\t\"frame\":{{\"x\":{0},\"y\":{1},\"w\":{2},\"h\":{3}}},",
                        rect.X, 
                        rect.Y,
                        rect.Width,
                        rect.Height));
                    //This tool does not rotate sprites
                    writer.WriteLine("\t\"rotated\":false,");
                    //Nor does it "trim" them to my knowledge
                    writer.WriteLine("\t\"trimmed\":false,");
                    writer.WriteLine(string.Format("\t\"spriteSourceSize\":{{\"x\":0,\"y\":0,\"w\":{0},\"y\":{1}}},",
                        rect.Width,
                        rect.Height));
                    writer.WriteLine(string.Format("\t\"sourceSize\":{{\"w\":{0},\"h\":{1}}}",
                        rect.Width,
                        rect.Height));

                    totalDictionaryCount++;
                    if(totalDictionaryCount < map.Count)
                    {
                         writer.WriteLine("},");
                    }
                    else
                    {
                         writer.WriteLine("}");
                    }
                }

                writer.WriteLine("}}");
            }
        }
    }
}

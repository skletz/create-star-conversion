module Utils
{
    export var toInt = (value: number): number =>
    {
        return ~~value;
    }
    
    export var getProportionalScale = (gameSize: number, assetSize: number): number =>
    {
        if (assetSize >= gameSize) return gameSize/assetSize;
        else return assetSize/gameSize;
    }

}
